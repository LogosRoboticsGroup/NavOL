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
print('---------------In domain-----------')
# srs = []
# spls = []
# for i in range(8):
#     if i <=1:
#         if args.extra is not None:
#             scene_dir = os.path.join(root, f'benchmark_in_domain_scene_{i:03d}_{args.extra}')
#         else:
#             scene_dir = os.path.join(root, f'benchmark_in_domain_scene_{i:03d}')
#     else:
#         scene_dir = os.path.join(root, f'benchmark_in_domain_scene_{i:03d}')
#     metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
#     min_sr = 100
#     min_spl = 100
#     for scene_dir in glob.glob(os.path.join(root, f'benchmark_out_domain_scene_{i:03d}*')):
#         with open(metric_file, 'r') as f:
#             lines = f.readlines()[1:]
#         assert len(lines) == 100, f"Expected 100 lines in {metric_file}, but got {len(lines)}"
#         sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
#         spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
#         if sr < min_sr:
#             min_sr = sr
#             min_spl = spl
#     srs.append(min_sr)
#     spls.append(min_spl)
#     print(f'benchmark_in_domain_scene_{i:03d} Success Rate: {min_sr*100:.1f}%, SPL: {min_spl*100:.1f}%')
# print(f'benchmark_in_domain_scene Overall Success Rate: {np.mean(srs)*100:.1f}%, Overall SPL: {np.mean(spls)*100:.1f}%')
# print('Scene 1 2 3 4 5 6 7 8 AVG')
# string = 'SR:   ' + ' & '.join([f'{srs[i]*100:.1f}' for i in range(8)]) + f' & {np.mean(srs)*100:.1f}' + '\n'
# string += 'SPL:  ' + ' & '.join([f'{spls[i]*100:.1f}' for i in range(8)]) + f' & {np.mean(spls)*100:.1f}' + '\n'
# string2 = ''
# for i in range(8):
#     string2 += f'{srs[i]*100:.1f} & {spls[i]*100:.1f} & '
# string2 += f'{np.mean(srs)*100:.1f} & {np.mean(spls)*100:.1f}\n'
# print(string)
# print(string2)

print('---------------Out domain-----------')
srs = []
spls = []
for i in range(8):
    scene_dir = os.path.join(root, f'benchmark_out_domain_scene_{i:03d}')
    metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
    with open(metric_file, 'r') as f:
        lines = f.readlines()[1:]
    assert len(lines) == 100, f"Expected 100 lines in {metric_file}, but got {len(lines)}"
    sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
    spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
    srs.append(sr)
    spls.append(spl)
    
    
    # if i <9:
    #     if args.extra is not None:
    #         scene_dir = os.path.join(root, f'benchmark_out_domain_scene_{i:03d}_{args.extra}')
    #     else:
    #         scene_dir = os.path.join(root, f'benchmark_out_domain_scene_{i:03d}')
    # else:
    #     scene_dir = os.path.join(root, f'benchmark_out_domain_scene_{i:03d}')
    # scene_dir = os.path.join(root, f'benchmark_out_domain_scene_{i:03d}')
    # metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
    # min_sr = 100
    # min_spl = 100
    # for scene_dir in glob.glob(os.path.join(root, f'benchmark_out_domain_scene_{i:03d}*')):
    #     # metric_file = os.path.join(scene_dir, 'metric_pointgoal_eval_distillation.csv')
    #     with open(metric_file, 'r') as f:
    #         lines = f.readlines()[1:]
    #     assert len(lines) == 100, f"Expected 100 lines in {metric_file}, but got {len(lines)}"
    #     sr = np.mean([float(line.strip().split(',')[0]) for line in lines])
    #     spl = np.mean([float(line.strip().split(',')[1]) for line in lines])
    #     if sr < min_sr:
    #         min_sr = sr
    #         min_spl = spl
    # srs.append(sr)
    # spls.append(spl)
    # print(f'benchmark_out_domain_scene_{i:03d} Success Rate: {min_sr*100:.1f}%, SPL: {min_spl*100:.1f}%')
print(f'benchmark_out_domain_scene Overall Success Rate: {np.mean(srs)*100:.1f}%, Overall SPL: {np.mean(spls)*100:.1f}%')
print('Scene 1 2 3 4 5 6 7 8 AVG')
string = 'SR:   ' + ' & '.join([f'{srs[i]*100:.1f}' for i in range(8)]) + f' & {np.mean(srs)*100:.1f}' + '\n'
string += 'SPL:  ' + ' & '.join([f'{spls[i]*100:.1f}' for i in range(8)]) + f' & {np.mean(spls)*100:.1f}' + '\n'
string2 = ''
for i in range(8):
    string2 += f'{srs[i]*100:.1f} & {spls[i]*100:.1f} & '
string2 += f'{np.mean(srs)*100:.1f} & {np.mean(spls)*100:.1f}\n'
print(string)
print(string2)