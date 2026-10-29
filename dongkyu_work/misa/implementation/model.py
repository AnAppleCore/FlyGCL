"""MISA downstream model adapter for FlyGCL.

Paper: Kang et al., ICLR 2025, Mask and Initial-Session Adaptation.
Upstream: https://github.com/kangzhiq/MISA (commit 72e7c960c73e).
This adapter reuses FlyGCL DualPrompt's compatible prompt forward/training
components; it does NOT implement ISA/FAM pretraining.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dualprompt import DualPrompt
from models.l2p import Prompt

logger = logging.getLogger(__name__)



class MISAExpertPrompt(Prompt):
    """Preserve upstream MISA's *local* topk prompt gather inside key slices.

    Official kangzhiq/MISA models/dualprompt.py::Prompt.forward computes
    cosine similarity with key[s:e], but gathers self.prompts with unshifted
    topk. Generic FlyGCL Prompt adds +s, changing the released baseline.
    Existing parameter objects and RNG initialization are preserved.
    """

    def __init__(self, previous: Prompt):
        nn.Module.__init__(self)
        self.pool_size = previous.pool_size
        self.selection_size = previous.selection_size
        self.prompt_len = previous.prompt_len
        self.dimention = previous.dimention
        self._diversed_selection = previous._diversed_selection
        self._batchwise_selection = previous._batchwise_selection
        self.key = previous.key
        self.prompts = previous.prompts
        self.register_buffer("frequency", previous.frequency)
        self.register_buffer("counter", previous.counter)
        self.last_topk = None
        self.last_selected_indices = None

    def forward(self, query: torch.Tensor, s=None, e=None, **kwargs):
        B, D = query.shape
        assert D == self.dimention, (
            f"Query dimention {D} does not match prompt dimention {self.dimention}"
        )
        if s is None and e is None:
            match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        else:
            assert s is not None and e is not None
            match = 1 - F.cosine_similarity(query.unsqueeze(1), self.key[s:e], dim=-1)
        if self.training and self._diversed_selection:
            scores = match * F.normalize(self.frequency, p=1, dim=-1)
        else:
            scores = match
        _, topk = scores.topk(self.selection_size, dim=-1, largest=False, sorted=True)
        if self._batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _, mosts = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().expand(B, -1)
        # Preserve released MISA behavior: local topk, NOT topk+s.
        self.counter += torch.bincount(
            topk.reshape(-1).clone(), minlength=self.pool_size
        )
        selected = self.prompts.repeat(B, 1, 1, 1).gather(
            1, topk.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, self.prompt_len, self.dimention
            ).clone()
        )
        self.last_topk = topk.detach()
        self.last_selected_indices = topk.detach()
        return match.gather(1, topk), selected


class MISA(DualPrompt):
    """Official downstream MISA configuration with pretrained ISA prompts.

    Upstream ``scripts/misa.sh`` invokes DualPrompt with ``--load_pt``.
    In FlyGCL MISA is a distinct model, so prompt loading is mandatory.
    """

    def __init__(self, backbone_name=None, num_classes=100, task_num=5,
                 misa_prompt_dir="./dongkyu_work/misa/checkpoints", **kwargs):
        if task_num not in (1, 2, 5, 10):
            raise ValueError(
                "MISA's official fixed 10-slot e-prompt pool requires "
                "a task count dividing 10 (1, 2, 5, or 10). "
                f"Got task_num={task_num}."
            )
        if kwargs.get("use_rp_gate") or kwargs.get("use_ema_head"):
            raise ValueError("MISA does not include FlyGCL RPFC or EMA head extensions.")
        if kwargs.get("step_num", -1) > 1:
            raise ValueError(
                "MISA follows official task boundaries; remove --step_num. "
                "Shared n_tasks protocol must be used."
            )
        if kwargs.get("mepo_backbone_path") or kwargs.get("cov_path"):
            raise ValueError("MISA cannot be combined with optional MePo calibration.")
        # The MISA method always uses ISA-initialized prompts; bypass the
        # legacy FlyGCL DualPrompt's hard-coded ./checkpoints loader.
        kwargs.pop("load_pt", None)
        super().__init__(
            backbone_name=backbone_name,
            num_classes=num_classes,
            task_num=task_num,
            pos_g_prompt=(0, 1),
            pos_e_prompt=(2, 3, 4),
            len_g_prompt=10,
            len_e_prompt=20,
            g_pool=1,
            e_pool=10,
            prompt_func="prompt_tuning",
            lambd=1.0,
            load_pt=False,
            **kwargs,
        )
        # MISA-only upstream prompt selection, no changes to other baselines.
        self.e_prompt = MISAExpertPrompt(self.e_prompt)
        # Avoid silently accepting an uninitialized model as MISA.
        self.misa_prompt_dir = str(misa_prompt_dir)
        self._load_misa_prompts(Path(self.misa_prompt_dir))

    @staticmethod
    def _read_prompt(path: Path, expected_shape):
        if not path.is_file():
            raise FileNotFoundError(
                f"MISA pretrained prompt not found: {path}. "
                "Get g_prompt.pt and e_prompt.pt from the official "
                "kangzhiq/MISA/pretrained_prompt directory."
            )
        # Official provided checkpoints are tensors; reject unexpected
        # pickled modules instead of loading arbitrary Python objects.
        tensor = torch.load(str(path), map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"MISA checkpoint {path} must contain a Tensor; got {type(tensor).__name__}."
            )
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"MISA checkpoint {path} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected_shape)}. Check the upstream weights "
                "and the fixed prompt configuration; do not tile prompt pools."
            )
        return tensor.detach().clone()

    def _load_misa_prompts(self, directory: Path):
        g_path = directory / "g_prompt.pt"
        e_path = directory / "e_prompt.pt"
        g_tensor = self._read_prompt(g_path, self.g_prompt.prompts.shape)
        e_tensor = self._read_prompt(e_path, self.e_prompt.prompts.shape)
        self.g_prompt.prompts = nn.Parameter(g_tensor)
        self.e_prompt.prompts = nn.Parameter(e_tensor)
        logger.info("Loaded MISA ISA-pretrained prompt tensors from %s", directory)
