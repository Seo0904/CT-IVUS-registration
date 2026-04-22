import os
from typing import Any, Dict, Optional


class WandbLogger:
    """Thin wrapper around wandb.

    - Safe no-op when wandb is unavailable or disabled.
    - Avoids importing wandb unless explicitly enabled.
    """

    def __init__(self, opt: Any):
        self.enabled = bool(getattr(opt, 'use_wandb', False))
        self._wandb = None
        self._run = None

        if not self.enabled:
            return

        try:
            import wandb  # type: ignore

            self._wandb = wandb
        except Exception as e:
            print(f"[wandb] disabled (import failed): {e}")
            self.enabled = False
            return

        project = getattr(opt, 'wandb_project', None) or 'WP-UNSB_ver2'
        entity = getattr(opt, 'wandb_entity', None)
        run_name = getattr(opt, 'wandb_run_name', None) or getattr(opt, 'name', None)
        group = getattr(opt, 'wandb_group', None)

        tags_raw = getattr(opt, 'wandb_tags', None)
        tags = None
        if isinstance(tags_raw, str) and tags_raw.strip():
            tags = [t.strip() for t in tags_raw.split(',') if t.strip()]

        mode = getattr(opt, 'wandb_mode', None)
        init_kwargs: Dict[str, Any] = {
            'project': project,
            'name': run_name,
            'config': {k: v for k, v in vars(opt).items() if self._is_jsonable(v)},
        }
        if entity:
            init_kwargs['entity'] = entity
        if group:
            init_kwargs['group'] = group
        if tags:
            init_kwargs['tags'] = tags
        if mode:
            init_kwargs['mode'] = mode

        # Allow env override (useful in containers)
        if os.environ.get('WANDB_MODE') and not mode:
            init_kwargs['mode'] = os.environ['WANDB_MODE']

        try:
            self._run = self._wandb.init(**init_kwargs)
        except Exception as e:
            print(f"[wandb] disabled (init failed): {e}")
            self.enabled = False
            return

        # Prefer epoch as x-axis for charts (while keeping internal step monotonic).
        self._define_default_metrics()

    def _define_default_metrics(self):
        if not self.enabled or self._wandb is None:
            return
        try:
            # A shared epoch key for train metrics and a dedicated epoch key for val metrics.
            try:
                self._wandb.define_metric('epoch', hidden=True)
                self._wandb.define_metric('val/epoch', hidden=True)
            except TypeError:
                self._wandb.define_metric('epoch')
                self._wandb.define_metric('val/epoch')
            self._wandb.define_metric('train/*', step_metric='epoch')
            self._wandb.define_metric('val/*', step_metric='val/epoch')
        except Exception as e:
            # don't crash training
            print(f"[wandb] define_metric failed: {e}")

    @staticmethod
    def _is_jsonable(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, (str, int, float, bool)):
            return True
        if isinstance(v, (list, tuple)):
            return all(WandbLogger._is_jsonable(x) for x in v)
        if isinstance(v, dict):
            return all(isinstance(k, str) and WandbLogger._is_jsonable(x) for k, x in v.items())
        return False

    @property
    def wandb(self):
        return self._wandb

    def log(self, data: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        if not self.enabled or self._wandb is None:
            return
        try:
            if step is None:
                self._wandb.log(data, commit=commit)
            else:
                self._wandb.log(data, step=step, commit=commit)
        except Exception as e:
            # don't crash training
            print(f"[wandb] log failed: {e}")

    def finish(self):
        if not self.enabled or self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass
