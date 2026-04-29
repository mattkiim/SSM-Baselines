## Safety-Gym:

Safety-Gym: 

- 'SafetyPointButton1-v0'  # 1e5
- 'SafetyCarButton1-v0'    # 1e5
- 'SafetyPointButton2-v0'  # 1e5
- 'SafetyPointPush1-v0'    # 1e5
- 'SafetyCarButton2-v0'    # 1e5

MOJOCO: 


### SSM

python examples/states/train_safe_matching_online.py \
  --wandb True \
  --project_name gymnasium_long \
  --run_name carbutton_ssm \
  --seed 0 \
  --env_name SafetyCarButton1-v0 \
  --max_steps 1000000 \
  --epoch_length 2000 \
  --start_training 10000 \
  --eval_interval 2000 \
  --log_interval 1000