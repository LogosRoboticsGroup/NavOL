#### in domain
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 0 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain \
    --save_path results/ablation_rollout_128_steps --use_mpc --extra mpc

CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 1 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain \
    --save_path results/ablation_rollout_128_steps --use_mpc --extra mpc

CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 2 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain \
    --save_path results/ablation_rollout_128_steps --use_mpc --extra mpc

CUDA_VISIBLE_DEVICES=3 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 3 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain \
    --save_path results/ablation_rollout_128_steps

CUDA_VISIBLE_DEVICES=4 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 4 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain \
    --save_path results/ablation_rollout_128_steps

CUDA_VISIBLE_DEVICES=5 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 5 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain \
    --save_path results/ablation_rollout_128_steps

CUDA_VISIBLE_DEVICES=6 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 6 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain \
    --save_path results/ablation_rollout_128_steps

CUDA_VISIBLE_DEVICES=7 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 7 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_in_domain/benchmark_in_domain \
    --save_path results/ablation_rollout_128_steps

#### out domain
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 0 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain \
    --save_path results/ablation_rollout_128_steps --extra mpc --max_linear_speed 0.7

CUDA_VISIBLE_DEVICES=1 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 1 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain \
    --save_path results/ablation_rollout_128_steps --extra mpc --max_linear_speed 0.7

CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 2 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain \
    --save_path results/ablation_rollout_128_steps --extra mpc --max_linear_speed 0.7

CUDA_VISIBLE_DEVICES=3 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 3 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain \
    --save_path results/ablation_rollout_128_steps --extra mpc --max_linear_speed 0.7

CUDA_VISIBLE_DEVICES=4 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 4 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain \
    --save_path results/ablation_rollout_128_steps --extra mpc --max_linear_speed 0.7

CUDA_VISIBLE_DEVICES=5 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 5 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain \
    --save_path results/ablation_rollout_128_steps --extra mpc --max_linear_speed 0.7

CUDA_VISIBLE_DEVICES=6 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 6 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain \
    --save_path results/ablation_rollout_128_steps --extra mpc --max_linear_speed 0.7

CUDA_VISIBLE_DEVICES=7 python scripts/rsl_rl/eval_navdp.py --headless --task pointgoal_eval_distillation \
    --num_envs 10 --scene_scale 1.0 --scene_index 7 --save \
    --checkpoint_path logs/rsl_rl/dingo_pointgoal_distillation/2025-11-11_19-00-26_rollout_128_steps/model_100_navdp.pt \
    --scene_dir source/scene_data_benchmark_scenes_out_domain/benchmark_out_domain \
    --save_path results/ablation_rollout_128_steps --extra mpc --max_linear_speed 0.7
