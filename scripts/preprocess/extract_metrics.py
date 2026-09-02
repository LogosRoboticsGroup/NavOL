
import os
import csv
import argparse

def compute_metrics(csv_path):
	success_list = []
	spl_list = []
	distance_list = []
	with open(csv_path, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			success_list.append(float(row['success']))
			spl_list.append(float(row['spl']))
			distance_list.append(float(row['distance']))
	n = len(success_list)
	if n == 0:
		return None
	return {
		'success': sum(success_list) / n,
		'spl': sum(spl_list) / n,
		'distance': sum(distance_list) / n,
        'n': n,
	}

def main():
    parser = argparse.ArgumentParser(description='Extract metrics from result folders.')
    parser.add_argument('--algorithm', type=str, default='navdp', help='Algorithm folder name under results/pointgoal')
    parser.add_argument('--scene', type=str, default='cluttered_easy', help='Scene folder name under results/pointgoal/{algorithm}')
    parser.add_argument('--task', type=str, default='pointgoal', help='task')
    args = parser.parse_args()

    base_dir = f"logs/rsl_rl/dingo_{args.task}"
    base_dir = "logs/rsl_rl/dingo_pointgoal"
    abs_base_dir = os.path.abspath(base_dir)
    if not os.path.exists(abs_base_dir):
        print(f"Base directory {abs_base_dir} does not exist.")
        return
    results = []
    for subdir in sorted(os.listdir(abs_base_dir)):
        subdir_path = os.path.join(abs_base_dir, subdir)
        if os.path.isdir(subdir_path):
            csv_path = os.path.join(subdir_path, f'metric_{args.task}.csv')
            if os.path.exists(csv_path):
                metrics = compute_metrics(csv_path)
                if metrics:
                    print(f"{subdir}: n={metrics['n']}, success={metrics['success']:.4f}, spl={metrics['spl']:.4f}, distance={metrics['distance']:.4f}")
                    results.append({
                        'subdir': subdir,
                        'n': metrics['n'],
                        'success': metrics['success'],
                        'spl': metrics['spl'],
                        'distance': metrics['distance']
                    })
                else:
                    print(f"{subdir}: No data in metric_{args.task}.csv")
            else:
                print(f"{subdir}: metric_{args.task}.csv not found")

    # 计算所有subdir的均值
    if results:
        mean_success = sum(r['success'] for r in results) / len(results)
        mean_spl = sum(r['spl'] for r in results) / len(results)
        mean_distance = sum(r['distance'] for r in results) / len(results)
        print(f"\nMean: success={mean_success:.4f}, spl={mean_spl:.4f}, distance={mean_distance:.4f}")

        # 保存到metrics_summary.csv
        summary_path = os.path.join(abs_base_dir, 'metrics_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['subdir', 'n', 'success', 'spl', 'distance'])
            for r in results:
                writer.writerow([r['subdir'], r['n'], f"{r['success']:.6f}", f"{r['spl']:.6f}", f"{r['distance']:.6f}"])
            writer.writerow(['mean', '', f"{mean_success:.6f}", f"{mean_spl:.6f}", f"{mean_distance:.6f}"])
        print(f"Results saved to {summary_path}")
    else:
        print("No valid metrics found.")

if __name__ == '__main__':
	main()
