import gc
import logging
from typing import Dict

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class FlyPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(FlyPrompt, self).__init__(*args, **kwargs)

        self.task_id = 0
        self.label_to_task: Dict[int, set] = {}
        self.head_snapshots = []

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

        # Optional MLP-based routing head training (independent from prompt branch)
        routing_mode = getattr(self.model_without_ddp, "routing_mode", "rpfc")
        if routing_mode == "mlp" and getattr(self.model_without_ddp, "router_mlp", None) is not None:
            with torch.no_grad():
                cls_features = self.model_without_ddp._forward_backbone_cls(x)
                rp_features = self.model_without_ddp._get_rp_features(cls_features)
            logits_gate = self.model_without_ddp.router_mlp(rp_features.detach())
            task_labels = torch.full((x.size(0),), self.task_id, device=self.device, dtype=torch.long)
            loss_gate = self.criterion(logits_gate, task_labels)
            loss = loss + loss_gate

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        self.model_without_ddp.update_ema_fc(expert_id=self.task_id)

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

                # use routing function (RPFC / random / KNN / NB / MLP) to get expert_ids
                expert_ids = self.model_without_ddp.route_experts(x, end=end)
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

    def oracle_evaluate(self, test_loader):
        """Oracle multi-task evaluation for FlyPrompt.

        For each class c we use up to the first two task ids where c has
        appeared during training. For a test sample, if any of those
        (prompt_t + g_t + EMA heads) combinations predicts correctly, it
        is counted as correct.
        """
        self.model_without_ddp.update()
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

                    # load snapshot of online head at task t
                    head = self.model_without_ddp.backbone.fc
                    snapshot = self.head_snapshots[t]
                    head.weight.data.copy_(snapshot["weight"].to(head.weight.device))
                    head.bias.data.copy_(snapshot["bias"].to(head.bias.device))

                    expert_ids = torch.full((x_sub.size(0),), t, device=self.device, dtype=torch.long)
                    logit_ls = self.model_without_ddp.forward_with_ema(x_sub, expert_ids=expert_ids)
                    logit_ls = [logit + self.mask for logit in logit_ls]
                    logit = self._ensemble_logits(logit_ls)

                    loss = self.criterion(logit, y_sub)
                    pred_sub = torch.argmax(logit, dim=-1)
                    correct_sub = (pred_sub == y_sub)
                    hit[idx_tensor] |= correct_sub

                    total_loss += loss.item()

                total_correct += hit.sum().item()
                total_num_data += batch_size

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
        # snapshot current classifier head g_t (online head)
        head = self.model_without_ddp.backbone.fc
        self.head_snapshots.append({
            "weight": head.weight.detach().cpu().clone(),
            "bias": head.bias.detach().cpu().clone(),
        })

        self.model_without_ddp.process_task_count()
        self.task_id += 1