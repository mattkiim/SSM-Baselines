#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot="${1:?Usage: bash scripts/train_rcrl_threshold10.sh ant|humanoid [training flags...]}"
shift

exec bash "$repo_root/scripts/train_rcrl_velocity.sh" "$robot" \
  --run_name="${robot}_rcrl_threshold10_3m" \
  --max_steps=3000000 \
  --replay_capacity=1000000 \
  --config.safety_threshold=10.0 \
  --eval_episodes=50 \
  --eval_max_episode_steps=1000 \
  "$@"
