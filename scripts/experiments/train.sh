python -m torch.distributed.run --nnodes=1 --nproc_per_node=8  scripts/rsl_rl/train_navdp.py --distributed --headless --task pointgoal_train_distillation --scene_scale 1.0 --rand_camera \
    agent.run_name=scene50_step8_mpc --scene_dir source/scene_data_3d_front_large2/3d_front_scene_50 --video_interval 1 --video --video_length 499 agent.save_interval=100 \
    agent.algorithm.num_learning_epochs=10 agent.algorithm.num_mini_batches=1 --num_envs 32 agent.num_steps_per_env=8 agent.algorithm.gradient_length=1 agent.algorithm.lambda_critic=0.1 \
    env.actions.joint_combined.navmesh_radius=0.25 env.actions.joint_combined.action_type=command env.actions.joint_combined.action_rand_p=0.8 agent.algorithm.learning_rate=1e-5 \
    env.actions.joint_combined.search_radius=0.1 env.actions.joint_combined.use_mpc=true > results/logs/scene50_step8_mpc.log 2>&1

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8  scripts/rsl_rl/train_navdp.py --distributed --headless --task pointgoal_train_distillation --scene_scale 1.0 --rand_camera \
    agent.run_name=scene50_step8 --scene_dir source/scene_data_3d_front_large2/3d_front_scene_50 --video_interval 1 --video --video_length 499 agent.save_interval=100 \
    agent.algorithm.num_learning_epochs=10 agent.algorithm.num_mini_batches=1 --num_envs 32 agent.num_steps_per_env=8 agent.algorithm.gradient_length=1 agent.algorithm.lambda_critic=0.1 \
    env.actions.joint_combined.navmesh_radius=0.25 env.actions.joint_combined.action_type=command env.actions.joint_combined.action_rand_p=0.8 agent.algorithm.learning_rate=1e-5 \
    env.actions.joint_combined.search_radius=0.1 > results/logs/scene50_step8.log 2>&1

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8  scripts/rsl_rl/train_navdp.py --distributed --headless --task pointgoal_train_distillation --scene_scale 1.0 --rand_camera \
    agent.run_name=scene50_step128_mpc --scene_dir source/scene_data_3d_front_large2/3d_front_scene_50 --video_interval 1 --video --video_length 499 agent.save_interval=100 \
    agent.algorithm.num_learning_epochs=10 agent.algorithm.num_mini_batches=16 --num_envs 32 agent.num_steps_per_env=128 agent.algorithm.gradient_length=1 agent.algorithm.lambda_critic=0.1 \
    env.actions.joint_combined.navmesh_radius=0.25 env.actions.joint_combined.action_type=command env.actions.joint_combined.action_rand_p=0.8 agent.algorithm.learning_rate=1e-5 \
    env.actions.joint_combined.search_radius=0.1 env.actions.joint_combined.use_mpc=true > results/logs/scene50_step128_mpc.log 2>&1

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8  scripts/rsl_rl/train_navdp.py --distributed --headless --task pointgoal_train_distillation --scene_scale 1.0 --rand_camera \
    agent.run_name=scene50_step128 --scene_dir source/scene_data_3d_front_large2/3d_front_scene_50 --video_interval 1 --video --video_length 499 agent.save_interval=100 \
    agent.algorithm.num_learning_epochs=10 agent.algorithm.num_mini_batches=16 --num_envs 32 agent.num_steps_per_env=128 agent.algorithm.gradient_length=1 agent.algorithm.lambda_critic=0.1 \
    env.actions.joint_combined.navmesh_radius=0.25 env.actions.joint_combined.action_type=command env.actions.joint_combined.action_rand_p=0.8 agent.algorithm.learning_rate=1e-5 \
    env.actions.joint_combined.search_radius=0.1 env.actions.joint_combined.use_mpc=true > results/logs/scene50_step128_mpc.log 2>&1

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8  scripts/rsl_rl/train_navdp.py --distributed --headless --task pointgoal_train_distillation --scene_scale 1.0 --rand_camera \
    agent.run_name=from_scratch --scene_dir source/scene_data_3d_front_large2/3d_front_scene_50 --video_interval 1 --video --video_length 499 agent.save_interval=100 \
    agent.algorithm.num_learning_epochs=10 agent.algorithm.num_mini_batches=16 --num_envs 32 agent.num_steps_per_env=128 agent.algorithm.gradient_length=1 agent.algorithm.lambda_critic=0.1 \
    env.actions.joint_combined.navmesh_radius=0.25 env.actions.joint_combined.action_type=command env.actions.joint_combined.action_rand_p=0.8 agent.algorithm.learning_rate=1e-5 \
    env.actions.joint_combined.search_radius=0.1 agent.policy.pretrained_model_path=null > results/logs/from_scratch.log 2>&1


    
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8  scripts/rsl_rl/train_navdp.py --distributed --headless --task pointgoal_train_distillation --scene_scale 1.0 --rand_camera \
    agent.run_name=scene50_step128_mpc --scene_dir source/scene_data_3d_front_large2/3d_front_scene_50 --video_interval 1 --video --video_length 499 agent.save_interval=100 \
    agent.algorithm.num_learning_epochs=10 agent.algorithm.num_mini_batches=16 --num_envs 32 agent.num_steps_per_env=128 agent.algorithm.gradient_length=1 agent.algorithm.lambda_critic=0.1 \
    env.actions.joint_combined.navmesh_radius=0.25 env.actions.joint_combined.action_type=command env.actions.joint_combined.action_rand_p=0.8 agent.algorithm.learning_rate=1e-5 \
    env.actions.joint_combined.search_radius=0.1 env.actions.joint_combined.use_mpc=true > results/logs/scene50_step128_mpc.log 2>&1

python -m torch.distributed.run --nnodes=1 --nproc_per_node=8  scripts/rsl_rl/train_navdp.py --distributed --headless --task pointgoal_train_distillation --scene_scale 1.0 --rand_camera \
    agent.run_name=scene50_step128_repeat2 --scene_dir source/scene_data_3d_front_large2/3d_front_scene_50 --video_interval 1 --video --video_length 499 agent.save_interval=100 \
    agent.algorithm.num_learning_epochs=10 agent.algorithm.num_mini_batches=16 --num_envs 32 agent.num_steps_per_env=128 agent.algorithm.gradient_length=1 agent.algorithm.lambda_critic=0.1 \
    env.actions.joint_combined.navmesh_radius=0.25 env.actions.joint_combined.action_type=command env.actions.joint_combined.action_rand_p=0.8 agent.algorithm.learning_rate=1e-5 \
    env.actions.joint_combined.search_radius=0.1 > results/logs/scene50_step128_repeat2.log 2>&1
