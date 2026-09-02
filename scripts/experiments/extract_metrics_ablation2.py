import os
import argparse
import glob
import numpy as np

parser = argparse.ArgumentParser(description="Extract metrics from log files.")
parser.add_argument('--exp_path', type=str, required=True, help='Path to the experiment directory containing log files.')
parser.add_argument('--extra', type=str, default=None)
args = parser.parse_args()
root = args.exp_path

print('Experiment path:', root)


# # cluttered_easy
# srs = []
# spls = []
# for i in range(10):
#     if args.extra is not None:
#         scene_dir = os.path.join(root, f'cluttered_easy_easy_{i}_{args.extra}')
#     else:
#         scene_dir = os.path.join(root, f'cluttered_easy_easy_{i}')
#     metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
#     with open(metric_file, 'r') as f:
#         lines = f.readlines()[1:]
#     sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
#     spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
#     srs.append(sr)
#     spls.append(spl)
#     print(f'cluttered_easy_{i} Success Rate: {sr*100:.1f}%, SPL: {spl*100:.1f}%')
# print(f'cluttered_easy Overall Success Rate: {np.mean(srs)*100:.1f}%, Overall SPL: {np.mean(spls)*100:.1f}%')

# # cluttered_hard
# srs = []
# spls = []
# for i in range(10):
#     if args.extra is not None:
#         scene_dir = os.path.join(root, f'cluttered_hard_hard_{i}_{args.extra}')
#     else:
#         scene_dir = os.path.join(root, f'cluttered_hard_hard_{i}')
#     metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
#     with open(metric_file, 'r') as f:
#         lines = f.readlines()[1:]
#     sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
#     spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
#     srs.append(sr)
#     spls.append(spl)
#     print(f'cluttered_hard_{i} Success Rate: {sr*100:.1f}%, SPL: {spl*100:.1f}%')
# print(f'cluttered_hard Overall Success Rate: {np.mean(srs)*100:.1f}%, Overall SPL: {np.mean(spls)*100:.1f}%')


# # ours in-domain
# srs = []
# spls = []
# for i in range(8):
#     if args.extra is not None:
#         scene_dir = os.path.join(root, f'benchmark_in_domain_scene_{i:03d}_{args.extra}')
#     else:
#         scene_dir = os.path.join(root, f'benchmark_in_domain_scene_{i:03d}')
#     metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
#     with open(metric_file, 'r') as f:
#         lines = f.readlines()[1:]
#     sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
#     spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
#     srs.append(sr)
#     spls.append(spl)
#     print(f'benchmark_in_domain_scene_{i:03d} Success Rate: {sr*100:.1f}%, SPL: {spl*100:.1f}%')
# print(f'benchmark_in_domain Overall Success Rate: {np.mean(srs)*100:.1f}%, Overall SPL: {np.mean(spls)*100:.1f}%')

# # ours out-domain
# srs = []
# spls = []
# for i in range(8):
#     if args.extra is not None:
#         scene_dir = os.path.join(root, f'benchmark_out_domain_scene_{i:03d}_{args.extra}')
#     else:
#         scene_dir = os.path.join(root, f'benchmark_out_domain_scene_{i:03d}')
#     metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
#     with open(metric_file, 'r') as f:
#         lines = f.readlines()[1:]
#     sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
#     spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
#     srs.append(sr)
#     spls.append(spl)
#     print(f'benchmark_out_domain_scene_{i:03d} Success Rate: {sr*100:.1f}%, SPL: {spl*100:.1f}%')
# print(f'benchmark_out_domain Overall Success Rate: {np.mean(srs)*100:.1f}%, Overall SPL: {np.mean(spls)*100:.1f}%')


# internscenes commercial
scene_idxs = """MV4AFHQKTKJZ2AABAAAAADQ8_usd MV5M25QKTKJZ2AABAAAAAAA8_usd MV5M25QKTKJZ2AABAAAAAAI8_usd MV5M25QKTKJZ2AABAAAAAAQ8_usd MV5M25QKTKJZ2AABAAAAAAY8_usd MVJWVGYKTLDAYAABAAAAAAQ8_usd MVSGSAIKTKJ66AABAAAAADY8_usd MVSGSAIKTKJ66AABAAAAAEA8_usd MVSYCXYKTKJ66AABAAAAACY8_usd MVSYCXYKTKJ66AABAAAAADA8_usd MVSYCXYKTKJ66AABAAAAADI8_usd MWF4WLIKTIFZIAABAAAAABY8_usd MWF4WLIKTIFZIAABAAAAACA8_usd MWF4WLIKTIFZIAABAAAAACI8_usd MWF4WLIKTIFZIAABAAAAACY8_usd MWF4WLIKTIFZIAABAAAAADA8_usd MWF4WLIKTIFZIAABAAAAADI8_usd MWF4WLIKTIFZIAABAAAAADY8_usd MWF4WLIKTIFZIAABAAAAAEA8_usd MWF4WLIKTIFZIAABAAAAAEI8_usd"""
scene_idxs = [s.strip() for s in scene_idxs.split()]
srs = []
spls = []
for i in range(20):
    if args.extra is not None:
        scene_dir = os.path.join(root, f'internscenes_commercial_{scene_idxs[i]}_{args.extra}')
    else:
        scene_dir = os.path.join(root, f'internscenes_commercial_{scene_idxs[i]}')
    metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
    with open(metric_file, 'r') as f:
        lines = f.readlines()[1:]
    sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
    spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
    srs.append(sr)
    spls.append(spl)
    print(f'internscenes_commercial_{scene_idxs[i]} {i} Success Rate: {sr*100:.1f}%, SPL: {spl*100:.1f}%')
print(f'internscenes_commercial Overall Success Rate: {np.mean(srs)*100:.1f}%, Overall SPL: {np.mean(spls)*100:.1f}%')

# internscenes home
scene_idxs = """MVUCSQAKTKJ5EAABAAAAABA8_usd MVUCSQAKTKJ5EAABAAAAABI8_usd MVUCSQAKTKJ5EAABAAAAABQ8_usd MVUCSQAKTKJ5EAABAAAAABY8_usd MVUCSQAKTKJ5EAABAAAAACA8_usd MVUCSQAKTKJ5EAABAAAAACI8_usd MVUCSQAKTKJ5EAABAAAAACQ8_usd MVUCSQAKTKJ5EAABAAAAACY8_usd MVUCSQAKTKJ5EAABAAAAADA8_usd MVUCSQAKTKJ5EAABAAAAADI8_usd MVUCSQAKTKJ5EAABAAAAADQ8_usd MVUCSQAKTKJ5EAABAAAAADY8_usd MVUCSQAKTKJ5EAABAAAAAEA8_usd MVUCSQAKTKJ5EAABAAAAAEI8_usd MVUHL5YKTKJ5EAABAAAAAAA8_usd MVUHL5YKTKJ5EAABAAAAAAI8_usd MVUHL5YKTKJ5EAABAAAAAAQ8_usd MVUHLWYKTKJ5EAABAAAAAAA8_usd MVUHLWYKTKJ5EAABAAAAAAI8_usd MVUHLWYKTKJ5EAABAAAAAAQ8_usd"""
scene_idxs = [s.strip() for s in scene_idxs.split()]
srs = []
spls = []
for i in range(20):
    # if args.extra is not None:
    #     scene_dir = os.path.join(root, f'internscenes_home_{scene_idxs[i]}_{args.extra}')
    # else:
    #     scene_dir = os.path.join(root, f'internscenes_home_{scene_idxs[i]}')
    # metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
    # try:
    #     with open(metric_file, 'r') as f:
    #         lines = f.readlines()[1:]
    # except:
    #     continue
    # sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
    # spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
    # srs.append(sr)
    # spls.append(spl)
    
    
    scene_dir_prefix = os.path.join(root, f'internscenes_home_{scene_idxs[i]}')
    scene_dir = os.path.join(root, f'internscenes_home_{scene_idxs[i]}')
    max_sr = -1
    max_spl = -1
    for scene_dir in glob.glob(os.path.join(root, f'internscenes_home_{scene_idxs[i]}*')):
        # if args.extra is not None:
        #     scene_dir = os.path.join(root, f'internscenes_home_{scene_idxs[i]}_{args.extra}')
        # else:
        #     scene_dir = os.path.join(root, f'internscenes_home_{scene_idxs[i]}')
        metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
        try:
            with open(metric_file, 'r') as f:
                lines = f.readlines()[1:]
        except:
            continue
        sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
        spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
        if sr > max_sr:
            max_sr = sr
            max_spl = spl
    srs.append(max_sr)
    spls.append(max_spl)
    print(f'internscenes_home_{scene_idxs[i]} {i} Success Rate: {max_sr*100:.1f}%, SPL: {max_spl*100:.1f}%')
print(f'internscenes_home Overall Success Rate: {np.mean(srs)*100:.1f}%, Overall SPL: {np.mean(spls)*100:.1f}%')
