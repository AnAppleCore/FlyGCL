"""FlyGCL shared Si-Blurry Trainer adapter for official SinglePrompt.

Keeps the source SinglePrompt train-time mini-batch-class mask by inheriting
DualPrompt's unmodified online_step / online_train / model_forward functions.
No expert/task routing and no global framework changes.
"""
import torch

from methods.dualprompt import DualPrompt


class SinglePromptTrainer(DualPrompt):
    def __init__(self, *args, **kwargs):
        if kwargs.get("step_num", -1) is not None and kwargs.get("step_num", -1) > 1:
            raise ValueError("SinglePrompt uses no internal task/step routing; omit --step_num")
        if kwargs.get("no_batchmask", False):
            raise ValueError("Official SinglePrompt requires per-batch train logit masking")
        if kwargs.get("use_rp_gate", False) or kwargs.get("use_ema_head", False):
            raise ValueError("SinglePrompt has no router or EMA head bank")
        mode = kwargs.get("singleprompt_eval_mode", "seen")
        if mode not in ("seen", "raw"):
            raise ValueError("singleprompt_eval_mode must be 'seen' or 'raw'")
        self.singleprompt_eval_mode = mode
        super().__init__(*args, **kwargs)

    def online_after_task(self, task_id):
        # Benchmark task boundaries control sample stream and evaluation only.
        # They must never select, add or reset SinglePrompt model parameters.
        self.task_id = task_id + 1

    def online_evaluate(self, test_loader, task_id=None, end=False):
        if self.singleprompt_eval_mode == "seen":
            return super().online_evaluate(test_loader, task_id=task_id, end=end)

        # Official SinglePrompt evaluates raw logits over all 100 outputs.
        # Reuse FlyGCL metrics and restore the train/evaluation mask afterwards.
        original_mask = self.mask
        try:
            self.mask = torch.zeros_like(original_mask)
            return super().online_evaluate(test_loader, task_id=task_id, end=end)
        finally:
            self.mask = original_mask
