d_real_losses = []
d_fake_losses = []
import re
import matplotlib.pyplot as plt
import os

log_path = 'data/experiment_result/UNSB/moving-mnist/20260214_184345/moving_mnist_paired_sb/loss_log.txt'
save_dir = '/workspace/data/experiment_result/UNSB/moving-mnist/20260214_184345'
os.makedirs(save_dir, exist_ok=True)

# ログから値を抽出
epochs = []
g_losses = []
d_real_losses = []
d_fake_losses = []
val_epochs = []
val_ssim = []
val_l2 = []
val_psnr = []

with open(log_path, 'r') as f:
    for line in f:
        # G loss, D loss（G: の値をg_lossとして取得）
        m = re.match(r'\(epoch: (\d+),.*D_real: ([\d\.-]+).*D_fake: ([\d\.-]+).*G: ([\d\.-]+)', line)
        if m:
            epoch = int(m.group(1))
            d_real = float(m.group(2))
            d_fake = float(m.group(3))
            g_loss = float(m.group(4))
            epochs.append(epoch)
            g_losses.append(g_loss)
            d_real_losses.append(d_real)
            d_fake_losses.append(d_fake)
        # val 指標
        m_val = re.match(r'\(epoch: (\d+),.*val_SSIM: ([\d\.-]+) val_L2: ([\d\.-]+) val_PSNR: ([\d\.-]+)', line)
        if m_val:
            val_epoch = int(m_val.group(1))
            val_epochs.append(val_epoch)
            val_ssim.append(float(m_val.group(2)))
            val_l2.append(float(m_val.group(3)))
            val_psnr.append(float(m_val.group(4)))


# デバッグ: 抽出データを表示
print('epochs:', epochs)
print('g_losses:', g_losses)
print('d_real_losses:', d_real_losses)
print('d_fake_losses:', d_fake_losses)

# G loss, D loss グラフ

# G loss グラフ
plt.figure(figsize=(10,5))
plt.plot(epochs, g_losses, label='G loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('G loss')
plt.legend()
plt.title('G loss')
plt.savefig(os.path.join(save_dir, 'G_loss_graph.png'))
plt.close()

# D_real loss グラフ
plt.figure(figsize=(10,5))
plt.plot(epochs, d_real_losses, label='D_real loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('D_real loss')
plt.legend()
plt.title('D_real loss')
plt.savefig(os.path.join(save_dir, 'D_real_loss_graph.png'))
plt.close()

# D_fake loss グラフ
plt.figure(figsize=(10,5))
plt.plot(epochs, d_fake_losses, label='D_fake loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('D_fake loss')
plt.legend()
plt.title('D_fake loss')
plt.savefig(os.path.join(save_dir, 'D_fake_loss_graph.png'))
plt.close()


# val SSIM グラフ
plt.figure(figsize=(10,5))
plt.plot(val_epochs, val_ssim, label='val SSIM', color='purple')
plt.xlabel('Epoch')
plt.ylabel('val SSIM')
plt.legend()
plt.title('val SSIM')
plt.savefig(os.path.join(save_dir, 'val_SSIM_graph.png'))
plt.close()

# val L2 グラフ
plt.figure(figsize=(10,5))
plt.plot(val_epochs, val_l2, label='val L2', color='orange')
plt.xlabel('Epoch')
plt.ylabel('val L2')
plt.legend()
plt.title('val L2')
plt.savefig(os.path.join(save_dir, 'val_L2_graph.png'))
plt.close()

# val PSNR グラフ
plt.figure(figsize=(10,5))
plt.plot(val_epochs, val_psnr, label='val PSNR', color='brown')
plt.xlabel('Epoch')
plt.ylabel('val PSNR')
plt.legend()
plt.title('val PSNR')
plt.savefig(os.path.join(save_dir, 'val_PSNR_graph.png'))
plt.close()

print('loss_graph.png, val_metrics_graph.png を出力しました')
