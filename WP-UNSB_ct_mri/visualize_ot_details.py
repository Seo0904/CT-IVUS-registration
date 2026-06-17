#!/usr/bin/env python3
import os
import argparse
import glob
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('matplotlib is required to run this script. Install it with pip install matplotlib')


def _is_none_array(arr):
    if arr is None:
        return True
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == np.object_:
        return arr.tolist() is None
    return False


def save_heatmap(array, path, title=None, xlabel=None, ylabel=None, cmap='viridis'):
    if array is None:
        return False
    array = np.asarray(array)
    if array.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(array, aspect='auto', origin='lower', cmap=cmap)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def save_curve(array, path, title=None, xlabel=None, ylabel=None):
    if array is None:
        return False
    array = np.asarray(array).reshape(-1)
    if array.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(array, marker='o' if array.size <= 50 else None)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def load_npz_value(npz, key):
    if key not in npz:
        return None
    try:
        value = npz[key]
    except Exception:
        return None
    return None if _is_none_array(value) else value


def visualize_file(npz_path, output_dir=None, overwrite=False):
    if output_dir is None:
        output_dir = os.path.dirname(npz_path)

    base = os.path.splitext(os.path.basename(npz_path))[0]
    with np.load(npz_path, allow_pickle=True) as npz:
        P = load_npz_value(npz, 'P')
        M = load_npz_value(npz, 'M')
        U = load_npz_value(npz, 'U')

    saved = []
    if M is not None:
        out_path = os.path.join(output_dir, f'{base}_M.png')
        if overwrite or not os.path.exists(out_path):
            if save_heatmap(M, out_path, title='OT cost matrix M', xlabel='valid tgt index', ylabel='fake frame index'):
                saved.append(out_path)
    if P is not None:
        out_path = os.path.join(output_dir, f'{base}_P.png')
        if overwrite or not os.path.exists(out_path):
            if save_heatmap(P, out_path, title='OT transport plan P', xlabel='valid tgt index', ylabel='fake frame index'):
                saved.append(out_path)
    if U is not None:
        out_path = os.path.join(output_dir, f'{base}_U.png')
        if overwrite or not os.path.exists(out_path):
            if save_curve(U, out_path, title='OT barycentric assignments U', xlabel='fake frame index', ylabel='tgt barycentric idx'):
                saved.append(out_path)

    return saved


def main():
    default_input = '/workspace/data/experiment_result/WP-UNSB_ver2/moving-mnist/20260328_231602/moving_mnist_seg_paired_sb_only_sOT/ot_details'
    parser = argparse.ArgumentParser(description='Visualize OT detail npz files saved by WP-UNSB_ver2.')
    parser.add_argument('input_dir', nargs='?', default=default_input, help='Directory containing ot_details .npz files')
    parser.add_argument('--pattern', default='*.npz', help='Glob pattern to match .npz files')
    parser.add_argument('--output_dir', default=None, help='Directory to save output PNG files (default: same as input_dir)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing PNG files')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        # None を渡すと visualize_file 側で npz と同じディレクトリに保存される
        output_dir = None

    npz_files = sorted(glob.glob(os.path.join(input_dir, '**', args.pattern), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f'No files found for pattern {args.pattern} in {input_dir} (searched recursively)')

    for npz_path in npz_files:
        saved = visualize_file(npz_path, output_dir=output_dir, overwrite=args.overwrite)
        if saved:
            print(f'Saved {len(saved)} files for {os.path.basename(npz_path)}:')
            for name in saved:
                print('  -', name)
        else:
            print(f'No visualization produced for {os.path.basename(npz_path)} (missing P/M/U or files exist).')


if __name__ == '__main__':
    main()
