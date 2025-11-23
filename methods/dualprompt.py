import gc
import logging
from typing import Dict

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class DualPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(DualPrompt, self).__init__(*args, **kwargs)

        self.task_id = 0
        self.label_to_task: Dict[int, set] = {}
        self.head_snapshots = []

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        self._collect_label_to_task(labels)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        del(images, labels)
        gc.collect()
        return _loss / _iter, _acc / _iter

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
        # Update EMA head bank to track the online classifier head
        if getattr(self.model_without_ddp, 'use_ema_head', False):
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

    def _collect_label_to_task(self, labels: torch.Tensor) -> None:
        """Update mapping from class label to the set of tasks where it appears.

        We store labels in the internal re-indexed space (0..len(exposed_classes)-1)
        consistent with how the classifier head is trained.
        """
        for j in range(len(labels)):
            cls = labels[j].item()
            # map to internal class index used in training
            if cls in self.exposed_classes:
                internal_id = self.exposed_classes.index(cls)
                if internal_id not in self.label_to_task:
                    self.label_to_task[internal_id] = set()
                self.label_to_task[internal_id].add(self.task_id)

    def oracle_evaluate(self, test_loader):
        """Oracle multi-task evaluation with per-task classifier snapshots.

        For each class c we use up to the first two task ids where c has
        appeared during training. For a test sample, if any of those
        (prompt_t + g_t) combinations predicts correctly, it is counted
        as correct.
        """
        self.model.eval()
        total_correct, total_num_data = 0.0, 0.0
        total_loss = 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)

        with torch.no_grad():
            for data in test_loader:
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                batch_size = y.size(0)
                hit = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

                # build mapping task_id -> indices in batch that use this task
                task_to_indices = {}
                for idx_in_batch in range(batch_size):
                    cls = int(y[idx_in_batch].item())
                    tasks = sorted(list(self.label_to_task.get(cls, [])))
                    if len(tasks) == 0:
                        continue
                    if len(tasks) > 2:
                        tasks = tasks[:2]
                    for t in tasks:
                        task_to_indices.setdefault(t, []).append(idx_in_batch)

                for t, indices in task_to_indices.items():
                    idx_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
                    x_sub = x[idx_tensor]
                    y_sub = y[idx_tensor]

                    # load snapshot of head g_t
                    head = self.model_without_ddp.backbone.fc
                    snapshot = self.head_snapshots[t]
                    head.weight.data.copy_(snapshot["weight"].to(head.weight.device))
                    head.bias.data.copy_(snapshot["bias"].to(head.bias.device))

                    logit = self.model_without_ddp.forward_with_task(x_sub, task_id=t)
                    logit = logit + self.mask
                    # we do not use loss meaningfully here; keep for completeness
                    loss = self.criterion(logit, y_sub)

                    pred_sub = torch.argmax(logit, dim=-1)
                    correct_sub = (pred_sub == y_sub)
                    hit[idx_tensor] |= correct_sub

                    total_loss += loss.item()

                total_correct += hit.sum().item()
                total_num_data += batch_size

                # build a pseudo prediction tensor for class-wise accuracy
                pred_full = y.clone()
                wrong_mask = ~hit
                if wrong_mask.any():
                    tmp = pred_full[wrong_mask].clone()
                    num_classes = len(self.exposed_classes)
                    pred_full[wrong_mask] = (tmp + 1) % max(num_classes, 2)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred_full)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

        avg_acc = total_correct / max(total_num_data, 1.0)
        avg_loss = total_loss / max(total_num_data, 1.0)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        return {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                if getattr(self.model_without_ddp, 'use_ema_head', False) and len(getattr(self.model_without_ddp, 'ema_heads', [])) > 0:
                    logit_ls = self.model_without_ddp.forward_with_ema(x)
                    logit_ls = [logit + self.mask for logit in logit_ls]
                    logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]
                    logit = torch.stack(logit_ls, dim=-1).max(dim=-1)[0]
                else:
                    logit = self.model(x)
                    logit = logit + self.mask
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

    def online_before_task(self, task_id):
        pass


    def online_after_task(self, cur_iter):
        # snapshot current classifier head g_t
        head = self.model_without_ddp.backbone.fc
        self.head_snapshots.append({
            "weight": head.weight.detach().cpu().clone(),
            "bias": head.bias.detach().cpu().clone(),
        })

        if not self.distributed:
            self.model.process_task_count()
        else:
            self.model.module.process_task_count()
        self.task_id += 1