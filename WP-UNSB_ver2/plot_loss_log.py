import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_metrics(line):
    # Extract all key:value pairs such as G: 9.431 or NCE_Y: 6.739
    pairs = re.findall(r'([A-Za-z0-9_]+):\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)', line)
    return {key: float(value) for key, value in pairs}


def plot_metric_epochs(metric_name, train_curve, val_curve, save_dir):
    """1つのメトリクに対して、epoch を横軸に train / val を重ねて描画する。"""

    plt.figure(figsize=(10, 5))

    # train 曲線
    if train_curve is not None:
        epochs, values = train_curve
        plt.plot(epochs, values, label='train')

    # val 曲線
    if val_curve is not None:
        epochs, values = val_curve
        plt.plot(epochs, values, label='val')

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(metric_name)
    if train_curve is not None or val_curve is not None:
        plt.legend()

    filename = f'{metric_name}.png'.replace(' ', '_')
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def sanitize_name(name):
    return name.replace(' ', '_').replace('/', '_').replace(':', '')


def main():
    parser = argparse.ArgumentParser(description='Plot training and validation metrics from a log file.')
    parser.add_argument('--exp_dir', default=None,
                        help='experiment directory that contains loss_log.txt (e.g. checkpoints/<name>/). If set, --log_path and --save_dir are auto-derived.')
    parser.add_argument('--log_path', default=None, help='path to loss_log.txt')
    parser.add_argument('--save_dir', default=None, help='directory to save plots (default: <exp_dir>/plots)')
    args = parser.parse_args()

    if args.exp_dir is not None:
        args.log_path = os.path.join(args.exp_dir, 'loss_log.txt')
        if args.save_dir is None:
            args.save_dir = os.path.join(args.exp_dir, 'plots')

    if args.log_path is None:
        raise ValueError('Either --exp_dir or --log_path must be provided.')
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(args.log_path), 'plots')

    os.makedirs(args.save_dir, exist_ok=True)

    # メトリクごと・epochごとの値を蓄積する
    # 例: train_metrics['G'][epoch] = [v1, v2, ...]
    train_metrics = defaultdict(lambda: defaultdict(list))
    val_metrics = defaultdict(lambda: defaultdict(list))

    with open(args.log_path, 'r') as f:
        for line in f:
            if '(epoch:' not in line:
                continue

            metrics = parse_metrics(line)
            if not metrics:
                continue

            epoch = None
            if 'epoch' in metrics:
                epoch = int(metrics.pop('epoch'))

            # 以前は iters/time/data をメタ情報として捨てていたが、
            # これらも含めて全部プロットできるように残しておく

            # validation 行かどうかを判定
            is_val = any(key.startswith('val_') for key in metrics) or ' val' in line

            if is_val:
                # val_ プレフィックスを外して集計
                for name, value in metrics.items():
                    if name.startswith('val_'):
                        base_name = name[4:]
                    else:
                        base_name = name
                    if epoch is not None:
                        val_metrics[base_name][epoch].append(value)
            else:
                for name, value in metrics.items():
                    if epoch is not None:
                        train_metrics[name][epoch].append(value)

    # epoch ごとに平均を取る
    def average_by_epoch(metric_epoch_values):
        curves = {}
        for name, epoch_values in metric_epoch_values.items():
            epochs = sorted(epoch_values.keys())
            values = [sum(vs) / len(vs) for e, vs in sorted(epoch_values.items())]
            curves[name] = (epochs, values)
        return curves

    train_curves = average_by_epoch(train_metrics)
    val_curves = average_by_epoch(val_metrics)

    # メトリク名の一覧（train と val の和集合）
    all_metric_names = sorted(set(list(train_curves.keys()) + list(val_curves.keys())))

    for name in all_metric_names:
        metric_name = sanitize_name(name)
        train_curve = train_curves.get(name)
        val_curve = val_curves.get(name)
        plot_metric_epochs(metric_name, train_curve, val_curve, args.save_dir)

    print(f'Saved plots to: {args.save_dir}')


if __name__ == '__main__':
    main()
