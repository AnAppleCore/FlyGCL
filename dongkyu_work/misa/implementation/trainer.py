"""FlyGCL MISA downstream trainer.

MISA retains the official downstream training and ISA prompts. Evaluation
applies FlyGCL's seen-class mask before loss and prediction for a common
comparison protocol. This deliberately differs from official MISA raw-logit
evaluation; the vendored official source remains unchanged.
"""

import torch

from methods.dualprompt import DualPrompt


class MISA(DualPrompt):
    """MISA downstream update and evaluation on the FlyGCL protocol."""

    def online_evaluate(self, test_loader, task_id=None, end=False):
        # Match FlyGCL DualPrompt evaluation: exclude unseen output classes
        # before computing both loss and predictions.
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)

        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                y = y.clone()
                for i in range(len(y)):
                    y[i] = self.exposed_classes.index(y[i].item())

                x = x.to(self.device)
                y = y.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(x)
                    logits = logits + self.mask
                    loss = self.criterion(logits, y)

                predictions = logits.argmax(dim=-1)
                topk_predictions = logits.topk(self.topk, 1, True, True).indices
                total_correct += (topk_predictions == y.unsqueeze(1)).sum().item()
                total_num_data += y.size(0)
                total_loss += loss.item()

                class_count, class_correct = self._interpret_pred(y, predictions)
                correct_l += class_correct.detach().cpu()
                num_data_l += class_count.detach().cpu()

        return {
            "avg_loss": total_loss / len(test_loader),
            "avg_acc": total_correct / total_num_data,
            "cls_acc": (correct_l / (num_data_l + 1e-5)).numpy().tolist(),
        }
