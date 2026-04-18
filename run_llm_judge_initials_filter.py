"""LLM-Judge pass on picked Initials synthesis records; drop low-quality ones.

Uses gpt-4o-mini (pointwise 3-dim: relevance / clarity / quality, 1-5 each)
and drops records with:
  - min(rel, cla, qua) <= min_dim_threshold
  - overall = (rel + cla + qua) / 3 < overall_threshold
  - or any judge error.

Output: filtered JSONL with a 'judge' field appended to each surviving record.
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

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


def parse_judge_output(raw: str):
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


async def judge_one(client, model, question, response, sem, max_retries=3):
    async with sem:
        delay = 2.0
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(question=question, response=response)},
                    ],
                    temperature=0,
                    max_tokens=200,
                )
                raw = resp.choices[0].message.content or ""
                parsed = parse_judge_output(raw)
                if parsed is not None:
                    return parsed
                # Unparseable; retry
            except (RateLimitError, APIError) as e:
                if attempt >= max_retries - 1:
                    return {"error": type(e).__name__, "message": str(e)[:300]}
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as e:
                return {"error": type(e).__name__, "message": str(e)[:300]}
        return {"error": "max_retries"}


async def main_async(args):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in in_path.open() if l.strip()]
    print(f"Loaded {len(recs)} records from {in_path}")

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [judge_one(client, args.model, r["prefix"], r["response"], sem) for r in recs]
    judges = await tqdm_asyncio.gather(*tasks, desc="judge")

    kept = []
    stats = {"n_input": len(recs), "n_error": 0, "n_dropped_min_dim": 0, "n_dropped_overall": 0, "n_kept": 0}
    min_dims = []
    overalls = []
    for r, j in zip(recs, judges):
        if not j or "error" in j:
            stats["n_error"] += 1
            continue
        md = min(j["relevance"], j["clarity"], j["quality"])
        ov = (j["relevance"] + j["clarity"] + j["quality"]) / 3
        min_dims.append(md)
        overalls.append(ov)
        if md <= args.min_dim_drop:
            stats["n_dropped_min_dim"] += 1
            continue
        if ov < args.overall_drop:
            stats["n_dropped_overall"] += 1
            continue
        r["judge"] = j
        kept.append(r)

    stats["n_kept"] = len(kept)
    if min_dims:
        stats["min_dim_mean"] = sum(min_dims) / len(min_dims)
    if overalls:
        stats["overall_mean"] = sum(overalls) / len(overalls)

    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats_path = out_path.with_suffix(".judge_stats.json")
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n{json.dumps(stats, indent=2)}")
    print(f"\nKept {len(kept)}/{len(recs)} -> {out_path}")
    print(f"Stats -> {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--min_dim_drop", type=int, default=3,
                        help="Drop records with min(rel, cla, qua) <= this (default 3 means drop min<=3)")
    parser.add_argument("--overall_drop", type=float, default=3.5,
                        help="Drop records with overall < this")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
