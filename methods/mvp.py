import copy
import datetime
import gc
import logging
import time
from typing import Dict

import torch
import torch.nn.functional as F

from methods._trainer import _Trainer

logger = logging.getLogger()


class MVP(_Trainer):
    def __init__(self, **kwargs):
        super(MVP, self).__init__(**kwargs)

        self.use_afs  = True
        self.use_mcr  = True
        self.use_mask = True

        self.alpha  = 0.5
        self.gamma  = 2.0
        self.margin  = 0.5

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

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct/total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            feature, mvp_mask = self.model_without_ddp.forward_features(x)
            logit = self.model_without_ddp.forward_head(feature)
            if mask is not None: # batchmask
                logit += mask
            elif self.use_mask:
                logit = logit * mvp_mask
                logit = logit + self.mask
            loss = self.loss_fn(feature, mvp_mask, y)
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
                    loss = F.cross_entropy(logit, y_sub)

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

                logit = self.model(x)
                logit = logit + self.mask
                loss = F.cross_entropy(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.mean().item()
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

    def _compute_grads(self, feature, y, mask):
        head = copy.deepcopy(self.model_without_ddp.backbone.fc)
        head.zero_grad()
        logit = head(feature.detach())
        if self.use_mask:
            logit = logit * mask.clone().detach()
        logit = logit + self.mask

        sample_loss = F.cross_entropy(logit, y, reduction='none')
        sample_grad = []
        for idx in range(len(y)):
            sample_loss[idx].backward(retain_graph=True)
            _g = head.weight.grad[y[idx]].clone()
            sample_grad.append(_g)
            head.zero_grad()
        sample_grad = torch.stack(sample_grad)    #B,dim

        head.zero_grad()
        batch_loss = F.cross_entropy(logit, y, reduction='mean')
        batch_loss.backward(retain_graph=True)
        total_batch_grad = head.weight.grad[:len(self.exposed_classes)].clone()  # C,dim
        idx = torch.arange(len(y))
        batch_grad = total_batch_grad[y[idx]]    #B,dim

        return sample_grad, batch_grad

    def _get_ignore(self, sample_grad, batch_grad):
        ign_score = (1. - torch.cosine_similarity(sample_grad, batch_grad, dim=1))#B
        return ign_score

    def _get_compensation(self, y, feat):
        head_w = self.model_without_ddp.backbone.fc.weight[y].clone().detach()
        cps_score = (1. - torch.cosine_similarity(head_w, feat, dim=1) + self.margin)#B
        return cps_score

    def _get_score(self, feat, y, mask):
        sample_grad, batch_grad = self._compute_grads(feat, y, mask)
        ign_score = self._get_ignore(sample_grad, batch_grad)
        cps_score = self._get_compensation(y, feat)
        return ign_score, cps_score

    def loss_fn(self, feature, mask, y):
        ign_score, cps_score = self._get_score(feature.detach(), y, mask)

        if self.use_afs:
            logit = self.model_without_ddp.forward_head(feature)
            logit = self.model_without_ddp.forward_head(feature / (cps_score.unsqueeze(1)))
        else:
            logit = self.model_without_ddp.forward_head(feature)
        if self.use_mask:
            logit = logit * mask
        logit = logit + self.mask
        log_p = F.log_softmax(logit, dim=1)
        loss = F.nll_loss(log_p, y)

        if self.use_mcr:
            loss = (1-self.alpha)* loss + self.alpha * (ign_score ** self.gamma) * loss
        return loss.mean() + self.model_without_ddp.get_similarity_loss()

    def report_training(self, sample_num, train_loss, train_acc):
        logger.info(
             f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
             f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
             f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
             f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))} | "
             f"N_Prompts {self.model_without_ddp.e_prompts.size(0)} | "
             f"N_Exposed {len(self.exposed_classes)} | "
             f"Counts {self.model_without_ddp.count.to(torch.int64).tolist()}"
             )