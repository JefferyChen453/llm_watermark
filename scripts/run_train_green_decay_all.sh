#!/bin/bash
# Launch all 3 green_loss_weight decay experiments sequentially.
# Each experiment lives on a separate branch of the verl submodule.
# After all runs complete, the submodule is restored to its original branch.
set -euo pipefail

VERL_DIR="/home/tianyichen/llm_watermark/verl"
ORIGINAL_BRANCH=$(git -C "$VERL_DIR" branch --show-current)
echo "[launcher] verl is currently on branch: $ORIGINAL_BRANCH"

cleanup() {
    echo "[launcher] Restoring verl to branch: $ORIGINAL_BRANCH"
    git -C "$VERL_DIR" checkout "$ORIGINAL_BRANCH"
}
trap cleanup EXIT

BRANCHES=(
    "green-decay-adaptive"
)
SCRIPTS=(
    "recipe/watermark_kd_ray/scripts/run_train_green_decay_adaptive.sh"
)

for i in "${!BRANCHES[@]}"; do
    branch="${BRANCHES[$i]}"
    script="${SCRIPTS[$i]}"

    echo ""
    echo "================================================================"
    echo "[launcher] Experiment $((i + 1))/3: branch=$branch"
    echo "================================================================"

    git -C "$VERL_DIR" checkout "$branch"
    echo "[launcher] Now on: $(git -C "$VERL_DIR" log --oneline -1)"

    # Run the training script from inside the verl directory
    # (recipe.watermark_kd_ray.main is relative to verl root)
    (cd "$VERL_DIR" && bash "$script")

    echo "[launcher] Finished experiment: $branch"
done

echo ""
echo "[launcher] All 3 experiments complete."
