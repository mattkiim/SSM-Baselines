#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot="${1:?Usage: bash scripts/train_rcrl_reachability.sh ant|humanoid|swimmer|hopper|cheetah|walker [flags...]}"
shift
exec bash "$repo_root/scripts/train_rcrl_velocity.sh" "$robot" \
  --config=examples/states/configs/rac_velocity_reachability_config.py \
  --run_name="${robot}_rcrl_reachability_3m" \
  --max_steps=3000000 --replay_capacity=1000000 \
  --eval_interval=30000 --save_interval=30000 \
  --eval_episodes=50 --eval_max_episode_steps=1000 --eval_seed=42 \
  --normalize_actions \
  "$@"
