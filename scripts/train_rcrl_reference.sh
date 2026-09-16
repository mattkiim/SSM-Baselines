#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robot="${1:?Usage: bash scripts/train_rcrl_reference.sh ant|humanoid|swimmer|hopper|cheetah|walker [flags...]}"
shift
exec bash "$repo_root/scripts/train_rcrl_reachability.sh" "$robot" \
  --config=examples/states/configs/rac_velocity_reference_config.py \
  --run_name="${robot}_rcrl_reference_3m" "$@"
