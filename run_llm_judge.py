#!/usr/bin/env python3
"""
LLM-as-Judge quality evaluation. Aligned with ICW paper (arXiv 2505.16934):
three dimensions — Relevance / Clarity / Quality — each rated 1-5, pointwise.

Special note: outputs were capped at 600 tokens by design, so the judge is
explicitly told NOT to penalize apparent truncation.

Usage:
    export OPENAI_API_KEY=...
    python run_llm_judge.py \
        --cells-file outputs/quality_eval/cells.json \
        --output-dir outputs/quality_eval/judge \
        --n-samples 200 \
        --model gpt-4o-mini \
        --concurrency 16
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI, RateLimitError, APIError
from tqdm.asyncio import tqdm_asyncio


SYSTEM_PROMPT = """You are a meticulous evaluator of AI-generated answers. You rate each response along three independent 1-5 dimensions:

- Relevance: does the response actually address the user's question? 5 = directly and completely answers what was asked; 1 = largely off-topic.
- Clarity: is the response well-organized, easy to read, and coherent? 5 = clean structure, no confusion; 1 = chaotic / incoherent.
- Quality: is the information substantive, accurate, and informative? 5 = rich, accurate, well-supported; 1 = vague, wrong, or empty.

IMPORTANT: The response was capped at 600 tokens by the generation system. Some answers may appear to end mid-thought — do NOT penalize apparent truncation. Judge only the quality of what was actually written, as if the visible portion were the intended answer.

Return ONLY a single JSON object with exactly these keys, no prose:
{"relevance": <1-5>, "clarity": <1-5>, "quality": <1-5>, "justification": "<one short sentence>"}"""


USER_TEMPLATE = """Question:
{question}

Response to evaluate:
{response}

Rate this response."""


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str) -> Optional[Dict]:
    """Extract JSON object from judge output; return None if malformed."""
    m = JSON_RE.search(raw)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    for k in ("relevance", "clarity", "quality"):
        if k not in d:
            return None
        try:
            d[k] = int(d[k])
        except (TypeError, ValueError):
            return None
        if not (1 <= d[k] <= 5):
            return None
    d.setdefault("justification", "")
    return d


async def judge_one(
    client: AsyncOpenAI,
    model: str,
    question: str,
    response: str,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> Optional[Dict]:
    user = USER_TEMPLATE.format(question=question, response=response)
    delay = 2.0
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or ""
                parsed = parse_judge_output(raw)
                if parsed is None:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    return {"error": "parse_failed", "raw": raw[:500]}
                return parsed
            except (RateLimitError, APIError) as e:
                if attempt >= max_retries - 1:
                    return {"error": str(type(e).__name__), "message": str(e)[:300]}
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as e:
                return {"error": str(type(e).__name__), "message": str(e)[:300]}
    return {"error": "max_retries"}


async def process_cell(
    client: AsyncOpenAI,
    model: str,
    input_path: Path,
    output_path: Path,
    n_samples: int,
    concurrency: int,
    cell_tag: str,
) -> Dict:
    records_all: List[Dict] = []
    with input_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records_all.append(json.loads(line))
    selected = records_all[:n_samples]

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        judge_one(client, model, r["prefix"], r["gen_completion"], sem)
        for r in selected
    ]
    results = await tqdm_asyncio.gather(*tasks, desc=f"judge {cell_tag}")

    out_records = []
    scores = {"relevance": [], "clarity": [], "quality": []}
    n_errors = 0
    for rec, judge in zip(selected, results):
        out = {
            "prefix": rec["prefix"],
            "gen_completion": rec["gen_completion"],
            "judge": judge,
        }
        out_records.append(out)
        if judge and "error" not in judge:
            for k in scores:
                scores[k].append(judge[k])
        else:
            n_errors += 1

    with output_path.open("w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def std(xs):
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    summary = {
        "cell_tag": cell_tag,
        "input_file": str(input_path),
        "judge_model": model,
        "n_samples": len(selected),
        "n_errors": n_errors,
        "n_scored": len(scores["relevance"]),
        "relevance_mean": mean(scores["relevance"]),
        "relevance_std": std(scores["relevance"]),
        "clarity_mean": mean(scores["clarity"]),
        "clarity_std": std(scores["clarity"]),
        "quality_mean": mean(scores["quality"]),
        "quality_std": std(scores["quality"]),
        "overall_mean": mean(
            [(a + b + c) / 3 for a, b, c in zip(scores["relevance"], scores["clarity"], scores["quality"])]
        ),
    }
    return summary


async def main_async(args):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(args.cells_file) as f:
        cells = json.load(f)
    print(f"Cells: {list(cells.keys())}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for tag, path in cells.items():
        summary_path = out_dir / f"{tag}_summary.json"
        out_path = out_dir / f"{tag}.jsonl"
        if summary_path.exists() and not args.overwrite:
            print(f"[skip] {tag} already done; delete to rerun")
            with summary_path.open() as f:
                all_summaries[tag] = json.load(f)
            continue

        summary = await process_cell(
            client=client,
            model=args.model,
            input_path=Path(path),
            output_path=out_path,
            n_samples=args.n_samples,
            concurrency=args.concurrency,
            cell_tag=tag,
        )
        with summary_path.open("w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        all_summaries[tag] = summary
        print(
            f"[{tag}] rel={summary['relevance_mean']:.2f} clarity={summary['clarity_mean']:.2f} "
            f"quality={summary['quality_mean']:.2f} err={summary['n_errors']}/{summary['n_samples']}"
        )

    with (out_dir / "all_cells_summary.json").open("w") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
