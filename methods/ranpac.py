import copy
import gc
import logging

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class RanPAC(_Trainer):
    def __init__(self, *args, **kwargs):
        super(RanPAC, self).__init__(*args, **kwargs)

        self.task_id = 0
        self.first_task_completed = False

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)

        if self.task_id == 0:
            return self._train_first_task(images, labels)
        else:
            return self._collect_features_for_statistics(images, labels)

    def _train_first_task(self, images, labels):
        _loss, _acc, _iter = 0.0, 0.0, 0

        # Train with multiple iterations as specified
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        self._collect_features_for_statistics(images, labels)

        return _loss / _iter, _acc / _iter

    def _collect_features_for_statistics(self, images, labels):
        images_copy = images.clone()
        labels_copy = labels.clone()
        
        # Map labels to exposed class indices
        for j in range(len(labels_copy)):
            labels_copy[j] = self.exposed_classes.index(labels_copy[j].item())

        images_copy = images_copy.to(self.device)
        labels_copy = labels_copy.to(self.device)

        images_copy = self.test_transform_tensor(images_copy)

        with torch.no_grad():
            self.model.eval()
            if self.distributed:
                self.model.module.collect_features_labels(images_copy, labels_copy)
            else:
                self.model.collect_features_labels(images_copy, labels_copy)

        del images_copy, labels_copy
        gc.collect()
        return 0.0, 0.0  # No training loss/acc for subsequent tasks

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

        if not end and self.task_id > 0:
            if self.distributed:
                saved_state = self.model.module.save_classifier_state()
            else:
                saved_state = self.model.save_classifier_state()

        if self.task_id > 0 and self.first_task_completed:
            if self.distributed:
                self.model.module.update_statistics_and_classifier()
            else:
                self.model.update_statistics_and_classifier()

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                
                # Map labels to exposed class indices
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

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

        if not end and self.task_id > 0:
            if self.distributed:
                self.model.module.restore_classifier_state(saved_state)
            else:
                self.model.restore_classifier_state(saved_state)

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def online_before_task(self, task_id):
        if task_id == 0:
            if not self.distributed:
                if self.model.use_g_prompt:
                    self.model.freeze_backbone_except_prompts()
                    logger.info("First task: g-prompts and classifier enabled for training")
                else:
                    self.model.freeze_backbone_except_adapters()
                    logger.info("First task: adapters and classifier enabled for training")
            else:
                if self.model.module.use_g_prompt:
                    self.model.module.freeze_backbone_except_prompts()
                    logger.info("First task: g-prompts and classifier enabled for training")
                else:
                    self.model.module.freeze_backbone_except_adapters()
                    logger.info("First task: adapters and classifier enabled for training")
        else:
            if not self.distributed:
                self.model.freeze_all_except_classifier()
            else:
                self.model.module.freeze_all_except_classifier()
            
            if not self.distributed:
                mode = "g-prompt" if self.model.use_g_prompt else "adapter"
            else:
                mode = "g-prompt" if self.model.module.use_g_prompt else "adapter"
            logger.info(f"Task {task_id} ({mode} mode): collecting features for RanPAC statistics")

    def online_after_task(self, cur_iter):
        if self.task_id == 0:
            logger.info("Completing first task training, setting up random projection")

            if not self.distributed:
                self.model.setup_rp()
                self.model.freeze_all_except_classifier()
            else:
                self.model.module.setup_rp()
                self.model.module.freeze_all_except_classifier()

            self.first_task_completed = True
            logger.info("Random projection initialized, adapters frozen")

        if not self.distributed:
            self.model.process_task_count()
        else:
            self.model.module.process_task_count()
        
        logger.info(f"Task {self.task_id} completed, statistics updated")
        self.task_id += 1