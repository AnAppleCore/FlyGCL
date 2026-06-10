import copy
import gc
import logging

import torch
import torch.nn.functional as F

from methods._trainer import _Trainer

logger = logging.getLogger()


class LwF(_Trainer):
    def __init__(self, *args, **kwargs):
        super(LwF, self).__init__(*args, **kwargs)
        self.task_id = 0
        self.teacher_model = None
        self.teacher_num_classes = 0

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
            student_logits = self.model(x)
            if mask is not None:
                train_logits = student_logits + mask
            else:
                train_logits = student_logits + self.mask
            ce_loss = self.criterion(train_logits, y)
            kd_loss = self._distillation_loss(x, student_logits)
            loss = ce_loss + float(self.lwf_lambda) * kd_loss
        return train_logits, loss

    def _distillation_loss(self, x, student_logits):
        if self.teacher_model is None or self.teacher_num_classes <= 0:
            return torch.zeros((), device=self.device)

        old_classes = min(self.teacher_num_classes, student_logits.size(1), len(self.exposed_classes))
        if old_classes <= 0:
            return torch.zeros((), device=self.device)

        teacher_device = next(self.teacher_model.parameters()).device
        with torch.no_grad():
            teacher_logits = self.teacher_model(x.to(teacher_device, non_blocking=True))
            teacher_logits = teacher_logits.to(student_logits.device, non_blocking=True)

        temperature = float(getattr(self, "lwf_temperature", 2.0))
        student_old = student_logits[:, :old_classes]
        teacher_old = teacher_logits[:, :old_classes]
        kd = F.kl_div(
            F.log_softmax(student_old / temperature, dim=1),
            F.softmax(teacher_old / temperature, dim=1),
            reduction="batchmean",
        )
        return kd * temperature * temperature

    @torch.no_grad()
    def _build_teacher(self):
        teacher = copy.deepcopy(self.model_without_ddp)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        if getattr(self, "lwf_teacher_on_gpu", True):
            teacher = teacher.to(self.device)
        else:
            teacher = teacher.to(torch.device("cpu"))
        self.teacher_model = teacher
        self.teacher_num_classes = len(self.exposed_classes)
        logger.info("[LwF] Updated frozen teacher with %s exposed classes", self.teacher_num_classes)

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
        self._build_teacher()
        self.model_without_ddp.process_task_count()
        self.task_id += 1
        if torch.cuda.is_available():
            logger.info("[LwF] CUDA max memory allocated: %.2f GB", torch.cuda.max_memory_allocated(self.device) / 1024 ** 3)
        gc.collect()
