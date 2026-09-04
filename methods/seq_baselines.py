import gc
import logging

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


_HEAD_PARAM_MARKERS = (
    "backbone.fc.",
    "backbone.head.",
    "backbone.classifier.",
    "fc.",
    "head.",
    "classifier.",
)


def is_classifier_head_param(name: str) -> bool:
    return any(marker in name for marker in _HEAD_PARAM_MARKERS)


class SeqFinetune(_Trainer):
    def __init__(self, *args, **kwargs):
        super(SeqFinetune, self).__init__(*args, **kwargs)
        self.task_id = 0

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
        logit, loss = self.model_forward(x, y, mask=None if self.no_batchmask else logit_mask)
        _, preds = logit.topk(self.topk, 1, True, True)

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
            loss = self.criterion(logit, y)
        return logit, loss

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

    def online_after_task(self, cur_iter):
        self.model_without_ddp.process_task_count()
        self.task_id += 1
        gc.collect()


class LinearProbe(SeqFinetune):
    def setup_distributed_model(self):
        super().setup_distributed_model()
        self._freeze_non_head_parameters()
        self.reset_opt()
        learnable = [name for name, param in self.model_without_ddp.named_parameters() if param.requires_grad]
        n_params = sum(param.numel() for param in self.model_without_ddp.parameters() if param.requires_grad)
        logger.info("[LinearProbe] Learnable Parameters after freezing:\t%s", n_params)
        logger.info("[LinearProbe] Learnable parameter names after freezing: %s", learnable)

    def _freeze_non_head_parameters(self):
        head_names = []
        for name, param in self.model_without_ddp.named_parameters():
            is_head = is_classifier_head_param(name)
            param.requires_grad = is_head
            if is_head:
                head_names.append(name)
        if not head_names:
            raise ValueError("LinearProbe could not find classifier head parameters to train.")


class SeqFinetuneSmallLR(SeqFinetune):
    pass
