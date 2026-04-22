"""
Test EMA implementation:
1. Load model as train
2. Run a few optimization steps with EMA
3. Compare: normal G output vs EMA G output
"""
import sys, os, torch, numpy as np
sys.path.insert(0, '/workspace/WP-UNSB')

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

sys.argv = [
    'train.py',
    '--dataroot', '/workspace/data/org_data/moving_mnist',
    '--dataroot_B', '/workspace/data/preprocessed/bspline_transformed',
    '--data_file_B', 'transformed_aligned.npy',
    '--name', 'moving_mnist_paired_sb',
    '--model', 'wpsb',
    '--dataset_mode', 'moving_mnist_paired',
    '--input_nc', '1', '--output_nc', '1',
    '--batch_size', '8',
    '--load_size', '64', '--crop_size', '64',
    '--num_frames_per_seq', '20',
    '--train_ratio', '0.7', '--val_ratio', '0.1',
    '--checkpoints_dir', '/workspace/data/experiment_result/WP-UNSB/moving-mnist/20260227_043347',
    '--display_id', '-1',
    '--lambda_SB', '1.0',
    '--lr', '1e-05',
    '--nce_idt', 'True',
    '--continue_train',
    '--epoch', 'latest',
    '--gpu_ids', '0',
    '--no_html',
    '--use_ema',
    '--ema_decay', '0.999',
]

opt = TrainOptions().parse()
dataset = create_dataset(opt)
model = create_model(opt)

data_iter = iter(dataset)
data = next(data_iter)

# Initialize 
model.data_dependent_initialize(data, None)
model.setup(opt)
model.parallelize()
model.init_ema()

# Check initial EMA state
print("\n=== Initial state (no optimization yet) ===")
model.set_input(data)
with torch.no_grad():
    model.forward()
fake_init = model.fake_B.detach().cpu()
mean_init = ((fake_init + 1) / 2 * 255).mean().item()
print(f"  Normal G forward: fake_B mean = {mean_init:.1f}")

# Apply EMA and check
model.ema_G.apply_shadow()
model.set_input(data)
with torch.no_grad():
    model.forward()
fake_ema_init = model.fake_B.detach().cpu()
mean_ema_init = ((fake_ema_init + 1) / 2 * 255).mean().item()
print(f"  EMA G forward:    fake_B mean = {mean_ema_init:.1f}")
model.ema_G.restore()

# Run training steps
print("\n=== Training with EMA updates ===")
for step in range(20):
    data = next(data_iter)
    model.set_input(data)
    model.optimize_parameters()  # This also updates EMA
    
    fake = model.fake_B.detach().cpu()
    mean_normal = ((fake + 1) / 2 * 255).mean().item()
    
    if step % 5 == 0 or step >= 15:
        # Check EMA output
        model.ema_G.apply_shadow()
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        fake_ema = model.fake_B.detach().cpu()
        mean_ema = ((fake_ema + 1) / 2 * 255).mean().item()
        model.ema_G.restore()
        
        print(f"  Step {step:2d}: Normal G = {mean_normal:.1f}, EMA G = {mean_ema:.1f}")

# Final comparison: save EMA weights temporarily and test
print("\n=== Save/Load EMA test ===")
tmp_dir = '/tmp/ema_test'
os.makedirs(tmp_dir, exist_ok=True)

model.save_dir = tmp_dir
model.save_networks('test')

# Check that EMA file was created
ema_file = os.path.join(tmp_dir, 'test_net_G_ema.pth')
g_file = os.path.join(tmp_dir, 'test_net_G.pth')
print(f"  G file exists: {os.path.exists(g_file)}")
print(f"  EMA file exists: {os.path.exists(ema_file)}")

# Load EMA into a fresh model check
if os.path.exists(ema_file):
    ema_state = torch.load(ema_file, map_location='cuda:0')
    
    # Load into netG module
    net = model.netG
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    net.load_state_dict(ema_state)
    
    # Forward with EMA weights
    model.set_input(data)
    with torch.no_grad():
        model.forward()
    fake_loaded = model.fake_B.detach().cpu()
    mean_loaded = ((fake_loaded + 1) / 2 * 255).mean().item()
    print(f"  Loaded EMA forward: fake_B mean = {mean_loaded:.1f}")

print("\nDone!")
