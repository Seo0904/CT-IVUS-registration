"""
EMA (Exponential Moving Average) utility for PyTorch models.

Usage:
    ema = EMA(model, decay=0.999)
    # In training loop:
    ema.update()                     # after optimizer.step()
    # For inference:
    ema.apply_shadow()               # swap EMA weights in
    model(input)                     # forward with EMA weights
    ema.restore()                    # swap original weights back
    # For saving:
    ema.save(path)                   # save EMA state dict
    ema.load(path, device)           # load EMA state dict
"""

import torch
import copy
from collections import OrderedDict


class EMA:
    """Exponential Moving Average of model parameters.

    Args:
        model: PyTorch model (can be DataParallel-wrapped)
        decay: EMA decay factor (default: 0.999)
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.shadow = OrderedDict()
        self.backup = OrderedDict()
        self._initialised = False

    # ---- core API -----------------------------------------------------------
    def register(self):
        """Copy current parameters as the initial shadow (EMA) weights."""
        module = self._unwrap(self.model)
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self._initialised = True

    def update(self):
        """Update shadow weights: shadow ← decay * shadow + (1 - decay) * param."""
        if not self._initialised:
            self.register()
        module = self._unwrap(self.model)
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self):
        """Swap model params with EMA (shadow) params. Call restore() to undo."""
        module = self._unwrap(self.model)
        for name, param in module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original (non-EMA) params after apply_shadow()."""
        module = self._unwrap(self.model)
        for name, param in module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    # ---- save / load --------------------------------------------------------
    def state_dict(self):
        """Return EMA shadow weights as a state dict (same keys as model)."""
        return OrderedDict((k, v.cpu().clone()) for k, v in self.shadow.items())

    def load_state_dict(self, state_dict, device=None):
        """Load EMA shadow weights."""
        self.shadow.clear()
        for k, v in state_dict.items():
            if device is not None:
                v = v.to(device)
            self.shadow[k] = v
        self._initialised = True

    def save(self, path):
        """Save EMA weights to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path, device=None):
        """Load EMA weights from disk."""
        sd = torch.load(path, map_location=device or "cpu")
        self.load_state_dict(sd, device)

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _unwrap(model):
        """Get the underlying module from DataParallel if needed."""
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        return model
