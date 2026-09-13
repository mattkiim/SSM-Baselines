import safety_gymnasium

def main():
    env = safety_gymnasium.make("SafetyCarButton1-v0", render_mode=None)
    print(f"[INFO] Environment type: {type(env)}")

    builder = env.env.env.env
    print(f"[DEBUG] Located builder: {type(builder)}")

    obs, info = env.reset()
    print("✅ Environment reset successfully.")

    # 构建完整物理世界
    builder._setup_simulation()
    task = builder.task
    if task.world is None:
        task._build()

    model = task.world.model
    print(f"[INFO] Original timestep: {model.opt.timestep}")
    model.opt.timestep = 0.01
    print(f"[INFO] Updated timestep to: {model.opt.timestep}")

    # ✅ 关键：重新 reset，让状态初始化
    obs, info = env.reset()
    print("✅ Environment reinitialized after timestep modification.")

    for step in range(10):
        action = env.action_space.sample()
        env.step(action)

    env.close()
    print("\n✅ Finished testing.")

if __name__ == "__main__":
    main()
