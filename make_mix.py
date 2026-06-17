import os
import numpy as np
import matplotlib.pyplot as plt

OUT = 'data/dust_box/mix'
os.makedirs(OUT, exist_ok=True)

# --- load ---
a = np.load('data/org_data/moving_mnist/mnist_test_seq.npy')   # (T, N, H, W) uint8 0-255
b = np.load('data/preprocessed/bspline_transformed/transformed_global.npy')  # (N, T, H, W) f64 0-1

SEQ = 1   # シーケンス2（0始まりで index 1）

# 最初のフレームだけを使う（1枚の 64x64 画像）
frameA = a[0, SEQ].astype(np.float32) / 255.0   # mnist シーケンス2 の先頭フレーム (64,64)
frameB = b[SEQ, 0].astype(np.float32)           # transformed_global シーケンス2 の先頭フレーム (64,64)
print('frameA', frameA.shape, frameA.min(), frameA.max())
print('frameB', frameB.shape, frameB.min(), frameB.max())

# --- UNSB-style time schedule ---
T = 10
tau = 0.01
np.random.seed(0)

incs = np.array([0] + [1.0 / (i + 1) for i in range(T - 1)])
times = np.cumsum(incs)
times = times / times[-1]
times = 0.5 * times[-1] + 0.5 * times
times = np.concatenate([np.zeros(1), times])
print('times =', np.round(times, 4))

# --- 漸化式 (1フレームに対して) ---
Xt_1 = frameB
Xt = frameA.copy()

def save(img, name):
    img = np.clip(img, 0, 1)
    plt.imsave(os.path.join(OUT, name), img, cmap='gray', vmin=0, vmax=1)

save(Xt, f'mix_00_t{times[0]:.3f}.png')
for t in range(1, T + 1):
    delta = times[t] - times[t - 1]
    denom = times[-1] - times[t - 1]
    inter = delta / denom
    scale = delta * (1 - delta / denom)

    noise = np.random.randn(*Xt.shape).astype(np.float32)
    Xt = (1 - inter) * Xt + inter * Xt_1 + np.sqrt(scale * tau) * noise
    save(Xt, f'mix_{t:02d}_t{times[t]:.3f}.png')

print('saved', T + 1, 'images to', OUT)
print(sorted(os.listdir(OUT)))
