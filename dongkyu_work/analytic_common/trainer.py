"""Analytic adapters using FlyGCL's unchanged stream, task loop and metric saving."""
import logging
import math
import torch
from torch import nn
from methods._trainer import _Trainer
from utils.train_utils import select_model

logger = logging.getLogger(__name__)


class AnalyticTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        repeats = kwargs.get('online_iter', 1)
        if not math.isfinite(repeats) or repeats < 1 or int(repeats) != repeats:
            raise ValueError('Analytic methods require a positive integer online_iter')
        if kwargs.get('topk', 1) < 1:
            raise ValueError('topk must be positive')
        for flag in ('use_amp', 'use_rp_gate', 'use_ema_head', 'use_mepo'):
            if kwargs.get(flag, False):
                raise ValueError(f'{flag} is not supported by analytic baselines')
        if (kwargs.get('step_num') or 0) > 1:
            raise ValueError('Analytic methods use shared benchmark task boundaries, not step_num')
        super().__init__(*args, **kwargs)
        if self.distributed:
            raise ValueError('Analytic statistics are single-device; set CUDA_VISIBLE_DEVICES to one GPU')
        self.optimizer = self.scheduler = None

    def run(self):
        if torch.cuda.device_count() == 0:
            raise RuntimeError('FlyGCL main.py requires a CUDA GPU. Use the analytic verification scripts for CPU checks.')
        return super().run()

    def setup_distributed_model(self):
        self.model = select_model(self.method, self.backbone, self.n_classes, self.n_tasks, self.kwargs).to(self.device)
        # Allocate after backbone placement, so --analytic_stats_device cpu works.
        self.model.initialize_statistics()
        self.model.eval()
        self.model_without_ddp = self.model
        self.criterion = nn.CrossEntropyLoss()
        logger.info('Analytic method=%s; stats=%s; online_iter=%s repeats feature/statistic updates; optimizer/lr unused',
                    self.method, self.model.stats_device_policy, self.online_iter)

    def update_schedule(self, reset=False):
        pass  # Closed-form/statistical updates have no optimizer or LR schedule.

    def _map_labels(self, labels):
        mapping = {label: i for i, label in enumerate(self.exposed_classes)}
        return torch.tensor([mapping[int(y)] for y in labels], device=self.device, dtype=torch.long)

    @torch.no_grad()
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        targets = self._map_labels(labels)
        loss_sum = acc_sum = 0.
        self.model.eval()
        for _ in range(int(self.online_iter)):
            augmented = self.train_transform(images.clone().to(self.device))
            features = self.model.expand(self.model.features(augmented))
            self.model.fit_features(features, targets)
            logits = self.model.scores(features, self.device, refresh=self.method != 'air') + self.mask
            loss_sum += self.criterion(logits, targets).item()
            preds = logits.topk(min(self.topk, len(self.exposed_classes)), 1).indices
            acc_sum += (preds == targets[:, None]).any(1).float().mean().item()
        return loss_sum / self.online_iter, acc_sum / self.online_iter

    @torch.no_grad()
    def online_evaluate(self, test_loader, task_id=None, end=False):
        self.model.eval()
        correct = total = batches = 0
        total_loss = 0.
        correct_l = torch.zeros(self.n_classes)
        count_l = torch.zeros(self.n_classes)
        for images, labels in test_loader:
            targets = self._map_labels(labels)
            logits = self.model(images.to(self.device)) + self.mask
            total_loss += self.criterion(logits, targets).item()
            pred = logits.argmax(1)
            top = logits.topk(min(self.topk, len(self.exposed_classes)), 1).indices
            correct += (top == targets[:, None]).any(1).sum().item()
            total += len(targets)
            batches += 1
            counts, hits = self._interpret_pred(targets, pred)
            count_l += counts.cpu()
            correct_l += hits.cpu()
        if not total:
            raise ValueError('No seen-class evaluation samples')
        # Match DualPrompt's mean-of-batch-means loss and per-class denominator.
        return dict(avg_loss=total_loss / batches, avg_acc=correct / total,
                    cls_acc=(correct_l / (count_l + 1e-5)).tolist())

    def online_before_task(self, task_id):
        self.task_id = task_id

    def online_after_task(self, task_id):
        self.task_id = task_id + 1  # Never reset accumulated statistics.

    def report_training(self, sample_num, train_loss, train_acc):
        logger.info('Train | Sample # %s | diagnostic_loss %.4f | train_acc %.4f | Num_Classes %s',
                    sample_num, train_loss, train_acc, len(self.exposed_classes))
