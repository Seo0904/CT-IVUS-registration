# check_pot_autograd_v2.py
import torch
import ot

def run_once(reg=0.05, iters=50, device=None, scale_M=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T, C, H, W = 20, 1, 64, 64

    fake = torch.randn(T, C, H, W, device=device, requires_grad=True)
    tgt  = torch.randn(T, C, H, W, device=device)

    M = torch.cdist(fake.reshape(T, -1), tgt.reshape(T, -1), p=2) ** 2  # (T,T)

    # 任意：Mのスケーリング（Noneならそのまま）
    if scale_M == "median":
        s = M.detach().median()
        M = M / (s + 1e-8)
    elif scale_M == "mean":
        s = M.detach().mean()
        M = M / (s + 1e-8)
    elif isinstance(scale_M, (float, int)):
        M = M / float(scale_M)

    print(f"\n=== reg={reg}, iters={iters}, scale_M={scale_M}, device={device} ===")
    print("M stats: min", M.min().item(), "median", M.median().item(), "mean", M.mean().item(), "max", M.max().item())

    a = torch.ones(T, device=device) / T
    b = torch.ones(T, device=device) / T

    cost = ot.sinkhorn2(a, b, M, reg=reg, numItermax=iters)
    print("type(cost):", type(cost))
    if torch.is_tensor(cost):
        print("cost:", cost.item(), "device:", cost.device, "dtype:", cost.dtype)
    else:
        print("cost (non-tensor):", cost)
        cost = torch.tensor(cost, device=device, dtype=fake.dtype)

    fake.grad = None
    cost.backward()

    gsum = float(fake.grad.abs().sum().item()) if fake.grad is not None else 0.0
    print("grad abs sum:", gsum)
    return gsum

if __name__ == "__main__":
    # まずあなたの設定
    run_once(reg=0.05, iters=50, scale_M=None)

    # regを大きく
    run_once(reg=5.0, iters=50, scale_M=None)
    run_once(reg=50.0, iters=50, scale_M=None)

    # Mを正規化して reg を戻す
    run_once(reg=0.05, iters=50, scale_M="median")
    run_once(reg=0.1,  iters=50, scale_M="median")