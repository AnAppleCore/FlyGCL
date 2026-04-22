import gc
import logging
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from methods._trainer import _Trainer

logger = logging.getLogger()


class FlyPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(FlyPrompt, self).__init__(*args, **kwargs)

        self.task_id = 0
        self.label_to_task: Dict[int, set] = {}

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        self.collect(images.clone(), labels.clone())

        # When step_num is explicitly set, advance the internal step schedule
        # by seen samples; otherwise we keep the original task-boundary flow.
        if getattr(self, "use_internal_step_schedule", False):
            batch_size_global = images.size(0) * self.world_size
            self._maybe_advance_internal_step(batch_size_global)

        del images, labels
        gc.collect()
        return _loss / _iter, _acc / _iter

    def collect(self, images, labels):
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())

        unique_labels = torch.unique(labels)
        for label in unique_labels:
            if label.item() not in self.label_to_task:
                self.label_to_task[label.item()] = set()
            self.label_to_task[label.item()].add(self.task_id)

        images = images.to(self.device)
        labels = labels.to(self.device)

        images = self.test_transform_tensor(images)

        with torch.no_grad():
            self.model.eval()
            self.model_without_ddp.collect(images, labels)

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)

        self.optimizer.zero_grad()
        if not self.no_batchmask:
            logit, loss = self.model_forward(x,y,mask=logit_mask)
        else:
            logit, loss = self.model_forward(x,y)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        if not getattr(self, "no_ema_ensemble", False):
            if hasattr(self.model_without_ddp, "update_ema_fc"):
                self.model_without_ddp.update_ema_fc()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model_without_ddp.update()

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit_raw = self.model_without_ddp.forward_with_rp(x)
                expert_ids = torch.argmax(logit_raw, dim=-1)

                if getattr(self, "no_ema_ensemble", False):
                    logit = self.model_without_ddp(x, expert_ids=expert_ids)
                    logit = logit + self.mask
                else:
                    logit_ls = self.model_without_ddp.forward_with_ema(x, expert_ids=expert_ids)
                    logit_ls = [logit + self.mask for logit in logit_ls]
                    logit = self._ensemble_logits(logit_ls)

                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def _ensemble_logits(self, logit_ls):
        if not hasattr(self, 'ensemble_method'):
            self.ensemble_method = "softmax_max_prob"

        if "softmax" in self.ensemble_method:
            logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]

        logit_stack = torch.stack(logit_ls, dim=-1)  # Shape: [batch_size, n_classes, n_experts]

        if "mean" in self.ensemble_method:
            return logit_stack.mean(dim=-1)
        elif "max_prob" in self.ensemble_method:
            return logit_stack.max(dim=-1)[0]
        elif "min_entropy" in self.ensemble_method:
            entropies = -torch.sum(logit_stack * torch.log(logit_stack + 1e-8), dim=1)  # [batch_size, n_experts]
            min_entropy_indices = torch.argmin(entropies, dim=-1)  # [batch_size]
            batch_indices = torch.arange(logit_stack.size(0), device=logit_stack.device)
            return logit_stack[batch_indices, :, min_entropy_indices]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        """Advance by task boundary unless explicit internal steps are used."""
        if not getattr(self, "use_internal_step_schedule", False):
            self.model_without_ddp.process_task_count()
        self.task_id += 1

    def analyze_expert_features(self):
        """Extract per-expert CLS features on the full test set, compute
        similarity and CKA matrices, and save them (plus heatmaps) under
        self.log_dir for the current seed.

        This is called once at the end of training from _Trainer.main_worker
        on the main process only.
        """
        if not hasattr(self, "model_without_ddp"):
            logger.warning("[FlyPrompt] model_without_ddp not found, skip expert analysis.")
            return

        model = self.model_without_ddp
        model.eval()

        if not hasattr(self, "test_dataset"):
            logger.warning("[FlyPrompt] test_dataset not found, skip expert analysis.")
            return

        device = self.device
        # Number of experts in the model may differ from benchmark n_tasks
        # when using internal step-based scheduling.
        n_experts = getattr(model, "task_num", self.n_tasks)

        # Build deterministic DataLoader over the full test set
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batchsize * 2,
            shuffle=False,
            num_workers=self.n_worker,
            pin_memory=True,
        )

        # Infer feature dimension using a small probe batch
        with torch.no_grad():
            sample_x, _ = next(iter(test_loader))
            sample_x = sample_x.to(device)
            # Use expert 0 just to probe the dimension
            probe_ids = torch.zeros(sample_x.size(0), dtype=torch.long, device=device)
            sample_feat = model.experts(model.backbone, sample_x, probe_ids)
            feat_dim = sample_feat.size(-1)

        num_samples = len(self.test_dataset)
        features = torch.zeros(n_experts, num_samples, feat_dim, dtype=torch.float32)
        common_features = torch.zeros(num_samples, feat_dim, dtype=torch.float32)

        logger.info(
            f"[FlyPrompt] Extracting CLS features for {n_experts} experts over "
            f"{num_samples} test samples (dim={feat_dim}) ..."
        )

        offset = 0
        with torch.no_grad():
            for batch in test_loader:
                x, _ = batch
                batch_size = x.size(0)
                x = x.to(device)

                idx_slice = slice(offset, offset + batch_size)

                # Common representation: frozen backbone CLS without any expert prompts.
                common_batch = model.backbone.forward_features(x)[:, 0]
                common_features[idx_slice, :] = common_batch.detach().cpu()

                for t in range(n_experts):
                    expert_ids = torch.full(
                        (batch_size,), t, dtype=torch.long, device=device
                    )
                    feat_t = model.experts(model.backbone, x, expert_ids)
                    features[t, idx_slice, :] = feat_t.detach().cpu()

                offset += batch_size

        # Save raw features for potential further analysis
        os.makedirs(self.log_dir, exist_ok=True)
        feat_path = os.path.join(self.log_dir, f"features_seed_{self.rnd_seed}.pt")
        torch.save(features, feat_path)
        logger.info(f"[FlyPrompt] Saved expert features to {feat_path}")

        # Also save the common (backbone-only) features used for residual analysis
        common_feat_path = os.path.join(
            self.log_dir, f"common_features_seed_{self.rnd_seed}.pt"
        )
        torch.save(common_features, common_feat_path)
        logger.info(
            f"[FlyPrompt] Saved common (backbone-only) features to {common_feat_path}"
        )

        # ---------- Cosine similarity between per-task mean features ----------
        mean_feats = features.mean(dim=1)  # [T, D]
        mean_norm = mean_feats / (mean_feats.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = mean_norm @ mean_norm.t()  # [T, T]

        sim_path = os.path.join(self.log_dir, f"similarity_seed_{self.rnd_seed}.npy")
        np.save(sim_path, sim_matrix.numpy())
        logger.info(f"[FlyPrompt] Saved expert similarity matrix to {sim_path}")

        # Plot heatmap for similarity matrix if matplotlib is available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(sim_matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("FlyPrompt expert similarity (cosine of mean CLS)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            sim_fig_path = os.path.join(
                self.log_dir, f"similarity_seed_{self.rnd_seed}.png"
            )
            fig.savefig(sim_fig_path)
            plt.close(fig)
            logger.info(f"[FlyPrompt] Saved similarity heatmap to {sim_fig_path}")
        except Exception as e:
            logger.exception(
                "[FlyPrompt] Failed to plot similarity heatmap: %s", e
            )

        # ---------- Linear CKA between expert representations ----------
        def _center_gram(x: torch.Tensor) -> torch.Tensor:
            # x: [N, D]
            n = x.size(0)
            unit = torch.ones((n, n), device=x.device)
            identity = torch.eye(n, device=x.device)
            h = identity - unit / n
            k = x @ x.t()
            return h @ k @ h

        def _cka(x: torch.Tensor, y: torch.Tensor) -> float:
            x = x - x.mean(0, keepdim=True)
            y = y - y.mean(0, keepdim=True)

            kx = _center_gram(x)
            ky = _center_gram(y)

            hsic = (kx * ky).sum()
            norm_x = torch.sqrt((kx * kx).sum() + 1e-8)
            norm_y = torch.sqrt((ky * ky).sum() + 1e-8)
            return (hsic / (norm_x * norm_y)).item()

        cka_matrix = torch.zeros(n_experts, n_experts, dtype=torch.float32)
        # For CKA we work on all CUB200 test samples (small enough).
        for i in range(n_experts):
            x = features[i]  # [N, D]
            for j in range(n_experts):
                y = features[j]
                cka_matrix[i, j] = _cka(x, y)

        cka_path = os.path.join(self.log_dir, f"cka_seed_{self.rnd_seed}.npy")
        np.save(cka_path, cka_matrix.numpy())
        logger.info(f"[FlyPrompt] Saved expert CKA matrix to {cka_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cka_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("FlyPrompt expert CKA")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            cka_fig_path = os.path.join(self.log_dir, f"cka_seed_{self.rnd_seed}.png")
            fig.savefig(cka_fig_path)
            plt.close(fig)
            logger.info(f"[FlyPrompt] Saved CKA heatmap to {cka_fig_path}")
        except Exception as e:
            logger.exception("[FlyPrompt] Failed to plot CKA heatmap: %s", e)

        # ---------- Residual expert analysis: (expert - mean over experts) ----------
        # Ref shape: [1, N, D], residual shape: [T, N, D]
        ref = features.mean(dim=0, keepdim=True)
        residual = features - ref

        residual_feat_path = os.path.join(
            self.log_dir, f"residual_features_seed_{self.rnd_seed}.pt"
        )
        torch.save(residual, residual_feat_path)
        logger.info(
            f"[FlyPrompt] Saved residual expert features to {residual_feat_path}"
        )

        # Cosine similarity between per-task mean residual features
        residual_mean = residual.mean(dim=1)  # [T, D]
        residual_mean_norm = residual_mean / (
            residual_mean.norm(dim=1, keepdim=True) + 1e-8
        )
        residual_sim_matrix = residual_mean_norm @ residual_mean_norm.t()

        residual_sim_path = os.path.join(
            self.log_dir, f"residual_similarity_seed_{self.rnd_seed}.npy"
        )
        np.save(residual_sim_path, residual_sim_matrix.numpy())
        logger.info(
            f"[FlyPrompt] Saved residual expert similarity matrix to {residual_sim_path}"
        )

        # Plot heatmap for residual similarity matrix if matplotlib is available
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(residual_sim_matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("FlyPrompt residual expert similarity (cosine of mean CLS)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            residual_sim_fig_path = os.path.join(
                self.log_dir, f"residual_similarity_seed_{self.rnd_seed}.png"
            )
            fig.savefig(residual_sim_fig_path)
            plt.close(fig)
            logger.info(
                f"[FlyPrompt] Saved residual similarity heatmap to {residual_sim_fig_path}"
            )
        except Exception as e:
            logger.exception(
                "[FlyPrompt] Failed to plot residual similarity heatmap: %s", e
            )

        # Linear CKA between residual expert representations
        residual_cka_matrix = torch.zeros(n_experts, n_experts, dtype=torch.float32)
        for i in range(n_experts):
            x_i = residual[i]
            for j in range(n_experts):
                y_j = residual[j]
                residual_cka_matrix[i, j] = _cka(x_i, y_j)

        residual_cka_path = os.path.join(
            self.log_dir, f"residual_cka_seed_{self.rnd_seed}.npy"
        )
        np.save(residual_cka_path, residual_cka_matrix.numpy())
        logger.info(
            f"[FlyPrompt] Saved residual expert CKA matrix to {residual_cka_path}"
        )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(residual_cka_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("Expert index")
            ax.set_title("FlyPrompt residual expert CKA")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()

            residual_cka_fig_path = os.path.join(
                self.log_dir, f"residual_cka_seed_{self.rnd_seed}.png"
            )
            fig.savefig(residual_cka_fig_path)
            plt.close(fig)
            logger.info(
                f"[FlyPrompt] Saved residual CKA heatmap to {residual_cka_fig_path}"
            )
        except Exception as e:
            logger.exception("[FlyPrompt] Failed to plot residual CKA heatmap: %s", e)

