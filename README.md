# Infra Notes

现在这个是jax框架下的Safe Score Matching算法，由QSM方法的Repo修改而来。

- jaxrl5/agents，里面有各个agent

- rac就是Reachability Constrained Reinforcement Learning论文对应的算法

- cal是Off-Policy Primal-Dual Safe Reinforcement Learning论文对应的算法

- examples/states，里面是各个agent在safe-gymnasium环境下面的训练接口
  - examples/states/configs，这里修改agent核心超参数
  - scripts/里的scripts/ssm.sh对应选择要train的safe-gymnasium task，比如，cargoal1。此外这里也可以改核心超参数，这里优先级最高。
  - 现在safe-gymnasium下配置好的agent有：train_score_matching_online（QSM）、train_safe_matching_online（SSM）、train_sac_lag_online、train_sac_cbf_online、train_rac_online（RCRL）、train_cal_online（CAL）。

- examples/quadrotor，里面是各个agent在toy case quadrotor环境下面的训练接口
  - examples/quadrotor/configs，这里修改agent核心超参数
  - scripts/里的scripts/quadrotor.sh对应选择要train的agent，比如，ssm。此外这里也可以改核心超参数，这里优先级最高。
  - 现在safe-gymnasium下配置好的agent有：train_safe_matching_online（SSM）、train_sac_lag_online、train_sac_cbf_online、train_rac_online（RCRL）、train_cal_online（CAL）。
  - 此外，还提供的一些组件方便进行可视化，有关如何运行可视化代码，可以参考scripts/里的scripts/Vh_visual.sh，这个就是能直接画出来之前展示的三张V_h的可视化。我现在把这三张也都上传了，位置在Vh_figures/中。
