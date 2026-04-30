import os
import numpy as np
import matplotlib.pyplot as plt


def load_and_select_first_sequence(path, desired_frames=20, seq_idx=1):
    a = np.load(path, allow_pickle=True)

    time_axis = None
    for i, s in enumerate(a.shape):
        if s == desired_frames:
            time_axis = i
            break

    if time_axis is None:
        if a.ndim >= 3 and a.shape[0] >= desired_frames:
            frames = a[:desired_frames]
        else:
            if a.ndim == 4:
                if a.shape[1] >= desired_frames:
                    frames = a[seq_idx, :desired_frames]
                else:
                    frames = np.moveaxis(a, 1, 0)
                    if frames.shape[0] >= desired_frames:
                        frames = frames[:desired_frames]
                    else:
                        pass
            else:
                frames = np.asarray(a)
    else:
        b = np.moveaxis(a, time_axis, 0)
        if b.ndim == 4:
            seq_idx = min(seq_idx, b.shape[1] - 1)
            frames = b[:, seq_idx]
        else:
            frames = b

    frames = np.asarray(frames)

    if frames.ndim == 4:
        frames = frames[:, 0]

    if frames.shape[0] < desired_frames:
        last = frames[-1]
        pads = np.repeat(last[np.newaxis], desired_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pads], axis=0)
    else:
        frames = frames[:desired_frames]

    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = frames[..., 0]

    return frames


def plot_four_rows(
    file_paths,
    out_dir='data/dust_box/bspline_transformed',
    out_name='moving_mnist_preview.png',
    seq_idx=1,
    num_frames=5
):
    row_titles = [
        'Original',
        'B-spline global 0.05',
        'B-spline global 0.1',
        'B-spline global 0.15',
    ]

    arrays = []
    for p in file_paths:
        if not os.path.exists(p):
            print('missing:', p)
            arrays.append(None)
            continue
        try:
            a = np.load(p, allow_pickle=True)
            print(p, 'loaded shape', a.shape)
            frames = load_and_select_first_sequence(p, desired_frames=20, seq_idx=seq_idx)
            arrays.append(frames)
        except Exception as e:
            print('failed load', p, e)
            arrays.append(None)

    rows = len(arrays)
    fig, axes = plt.subplots(rows, num_frames, figsize=(2.2 * num_frames, 2.0 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, out_name)

    for i, frames in enumerate(arrays):
        for t in range(num_frames):
            ax = axes[i, t]
            ax.axis('off')

            if i == 0:
                ax.set_title(f'Frame {t}', fontsize=10)

            if t == 0:
                ax.text(
                    -0.15, 0.5, row_titles[i],
                    transform=ax.transAxes,
                    fontsize=11,
                    va='center',
                    ha='right'
                )

            if frames is None:
                ax.set_facecolor('black')
                continue

            im = frames[t]
            if im.ndim == 3 and im.shape[2] == 1:
                im = im[..., 0]

            if im.ndim == 3 and im.shape[2] == 3:
                ax.imshow(im.astype(np.uint8))
            else:
                ax.imshow(im, cmap='gray')

    plt.subplots_adjust(wspace=0.05, hspace=0.35)
    plt.savefig(out_png, dpi=200, bbox_inches='tight', pad_inches=0.1)
    print('saved', out_png)

    try:
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    files = [
        'data/org_data/moving_mnist/mnist_test_seq.npy',
        'data/preprocessed/bspline_transformed/transformed_global_0.05_3.npy'
        'data/preprocessed/bspline_transformed/transformed_global_0.1_3.npy',
        'data/preprocessed/bspline_transformed/transformed_global.npy',
    ]
    plot_four_rows(files, seq_idx=1, num_frames=5)