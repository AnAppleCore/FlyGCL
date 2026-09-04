import gc
import logging

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class EWC(_Trainer):
    def __init__(self, *args, **kwargs):
        super(EWC, self).__init__(*args, **kwargs)
        self.task_id = 0
        self.ewc_fisher = None
        self.ewc_params = None
        self._fisher_param_names = []
        self._cur_fisher = None
        self._cur_fisher_seen = 0

    def _named_trainable_parameters(self):
        return [(name, param) for name, param in self.model_without_ddp.named_parameters() if param.requires_grad]

    def _state_device(self):
        return self.device if getattr(self, "ewc_fisher_on_gpu", True) else torch.device("cpu")

    def _ensure_current_fisher(self):
        if self._cur_fisher is not None:
            return
        fisher_device = self._state_device()
        self._cur_fisher = {
            name: torch.zeros_like(param, device=fisher_device)
            for name, param in self._named_trainable_parameters()
        }
        self._cur_fisher_seen = 0

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        del images, labels
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

        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        x = self.train_transform(x)

        self.optimizer.zero_grad(set_to_none=True)
        logit, loss, ce_loss = self.model_forward(x, y, mask=None if self.no_batchmask else logit_mask)
        _, preds = logit.topk(self.topk, 1, True, True)

        use_empirical = bool(getattr(self, "ewc_empirical_labels", True))
        fisher_loss = ce_loss if use_empirical else self.criterion(logit, torch.argmax(logit.detach(), dim=-1))
        fisher_loss.backward(retain_graph=True)
        self._accumulate_online_fisher(batch_size=y.size(0))

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)
        return total_loss, total_correct / total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit = logit + mask
            else:
                logit = logit + self.mask
            ce_loss = self.criterion(logit, y)
            loss = ce_loss + self._ewc_penalty()
        return logit, loss, ce_loss

    def _ewc_penalty(self):
        if self.ewc_fisher is None or self.ewc_params is None:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        param_dict = dict(self.model_without_ddp.named_parameters())
        for name in self._fisher_param_names:
            param = param_dict.get(name)
            if param is None or not param.requires_grad:
                continue
            fisher = self.ewc_fisher[name].to(param.device, non_blocking=True)
            old_param = self.ewc_params[name].to(param.device, non_blocking=True)
            penalty = penalty + (fisher * (param - old_param).pow(2)).sum()
        return 0.5 * float(self.ewc_lambda) * penalty

    @torch.no_grad()
    def _snapshot_parameters(self):
        device = self._state_device()
        snapshot = {}
        for name, param in self._named_trainable_parameters():
            snapshot[name] = param.detach().clone().to(device)
        return snapshot

    @torch.no_grad()
    def _accumulate_online_fisher(self, batch_size: int):
        self._ensure_current_fisher()
        if batch_size <= 0:
            return
        fisher_device = self._state_device()
        for name, param in self._named_trainable_parameters():
            if param.grad is not None:
                self._cur_fisher[name].add_(param.grad.detach().pow(2).to(fisher_device), alpha=batch_size)
        self._cur_fisher_seen += int(batch_size)

    @torch.no_grad()
    def _finalize_current_fisher(self):
        self._ensure_current_fisher()
        if self._cur_fisher_seen <= 0:
            logger.warning("[EWC] No online gradients accumulated for task %s", self.task_id)
            return {name: value.detach().clone() for name, value in self._cur_fisher.items()}
        return {
            name: value.detach().clone().div(float(self._cur_fisher_seen))
            for name, value in self._cur_fisher.items()
        }

    @torch.no_grad()
    def _reset_current_fisher(self):
        self._cur_fisher = None
        self._cur_fisher_seen = 0

    @torch.no_grad()
    def _merge_fisher(self, task_fisher):
        gamma = float(getattr(self, "ewc_gamma", 1.0))
        if self.ewc_fisher is None:
            self.ewc_fisher = {name: value.detach().clone() for name, value in task_fisher.items()}
            self._fisher_param_names = list(self.ewc_fisher.keys())
            return
        for name, value in task_fisher.items():
            if name not in self.ewc_fisher:
                self.ewc_fisher[name] = value.detach().clone()
            else:
                self.ewc_fisher[name].mul_(gamma).add_(value)
        self._fisher_param_names = list(self.ewc_fisher.keys())

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)

        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logit = self.model(x) + self.mask
                    loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()
                total_loss += loss.item()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        return {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

    def online_before_task(self, task_id):
        self.task_id = task_id
        self._reset_current_fisher()

    def online_after_task(self, cur_iter):
        logger.info("[EWC] Consolidating online Fisher for task %s from %s samples", cur_iter, self._cur_fisher_seen)
        task_fisher = self._finalize_current_fisher()
        self._merge_fisher(task_fisher)
        self.ewc_params = self._snapshot_parameters()
        self._reset_current_fisher()
        self.model_without_ddp.process_task_count()
        self.task_id += 1
        if torch.cuda.is_available():
            logger.info("[EWC] CUDA max memory allocated: %.2f GB", torch.cuda.max_memory_allocated(self.device) / 1024 ** 3)
        gc.collect()
