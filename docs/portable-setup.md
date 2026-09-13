# Run on another machine

Use Python 3.10 on Linux. Run commands from the repository root. The working
experiment used an NVIDIA A40; CPU is sufficient for tests but slower for training.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Choose one:
python -m pip install -r requirements-rcrl.txt         # CPU
# python -m pip install -r requirements-rcrl-cuda12.txt # NVIDIA CUDA 12
python -m pip install --no-deps -r requirements-rcrl-legacy.txt
python -m pip install -e .
```

The RCRL requirements pin the core versions installed on the training machine;
they are not a complete lockfile. The separate legacy install is intentional:
`gymnasium-robotics==1.2.2` declares `numpy<1.24`, whereas this JAX version
requires `numpy>=1.26`. We retain the versions used for these velocity tasks
and install the two legacy simulator packages without dependency resolution.
`pip check` therefore reports the known robotics/NumPy mismatch. This setup
is scoped to the tested velocity tasks, not all legacy robotics environments.
CUDA wheels require a compatible NVIDIA driver
on the host. MuJoCo/GLFW require system OpenGL libraries. If pygame 2.1.0 builds
from source, install your distribution's SDL2 development libraries first.
The old `requirements.txt` is retained for historical, broader examples; use the
RCRL-specific files above for these experiments.

Verify the backend and run the focused checks:

```bash
python -c 'import jax; print(jax.devices())'
JAX_PLATFORMS=cpu python -m unittest tests.test_reachability tests.test_rcrl_reference tests.test_evaluation
```

Run a short pipeline check before allocating a full run:

```bash
JAX_PLATFORMS=cpu bash scripts/train_rcrl_reference.sh ant \
  --run_name=ant_reference_smoke --max_steps=512 --start_training=256 \
  --replay_capacity=1024 --eval_interval=512 --save_interval=512 \
  --eval_episodes=2 --eval_max_episode_steps=100 --log_interval=32 --notqdm
```

For full seed-0 runs with periodic evaluation:

```bash
bash scripts/train_rcrl_reference.sh ant
bash scripts/train_rcrl_reference.sh humanoid
```

The launcher defaults to CUDA and uses `python` from the active environment.
Override with `JAX_PLATFORMS=cpu` or `RCRL_PYTHON=/path/to/python` as needed.
See [the experiment protocol](rcrl-reference-velocity.md) for settings and the
remaining differences from the original RCRL implementation. Use separate tmux
sessions for concurrent jobs that should survive SSH disconnection.

## What transfers

Git contains source, tests, documentation, and small configuration assets.
Checkpoints, logs, evaluation outputs, local environments, supplied reference
material, and nested reference repositories are excluded. Transfer selected
`results/...` directories separately if you need trained policies: retain their
`config.json` alongside the checkpoints. A code clone starts fresh training.

## Optional reference repositories

The RCRL training code does not require either cloned repository. To inspect the
reference used in our comparison:

```bash
git clone https://github.com/mahaitongdae/Reachability_Constrained_RL.git
git -C Reachability_Constrained_RL checkout 83993a775081359275bae6dc77b28b0deb5bb499
```

Our RESPO bridge and callback changes are preserved as a patch against the
existing RESPO fork. Clone access may require your GitHub SSH credentials:

```bash
git clone git@github.com:mattkiim/respo-clone.git respo-clone
git -C respo-clone checkout da0b718b274a2340d0d62fcb69f0005d894a9ca8
git -C respo-clone apply --check ../patches/respo-velocity.patch
git -C respo-clone apply ../patches/respo-velocity.patch
JAX_PLATFORMS=cpu python -m unittest tests.test_respo_velocity
```

The RESPO bridge test skips when this optional checkout is absent. For RESPO
training dependencies and commands see [the benchmark notes](reachability-benchmark.md).

## Validation performed for this commit

The CPU requirements and legacy simulator packages were installed into a fresh
Python 3.10 virtual environment. A checkout containing only staged files passed
the RCRL/evaluation suite (13 tests passed; the optional RESPO test skipped).
The documented 512-step Ant smoke run completed training, evaluation and
checkpoint saving from that checkout using the active environment's Python.
The full workspace also passed all 14 tests with its optional RESPO checkout.
The RESPO patch was applied to its recorded base and compared byte-for-byte
with the three adapted source files. CUDA installation was not repeated in the
fresh environment; previous full Ant/Humanoid runs used the recorded GPU stack.
