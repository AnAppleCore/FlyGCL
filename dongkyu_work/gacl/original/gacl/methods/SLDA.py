# The codes in this file was adopted from https://github.com/tyler-hayes/Deep_SLDA/blob/master/experiment.py
import time
import datetime
import torch
import sys
from torch.utils.data import DataLoader
from methods._trainer import _Trainer
from utils.online_sampler import OnlineTestSampler


def pool_feat(features):
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
    features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)  # mb x num_channels
    return feat

class SLDA(_Trainer):
    def __init__(self, **kwargs):
        super(SLDA, self).__init__(**kwargs)
        
        self.shrinkage_param = kwargs.get("shrinkage")
        self.streaming_update_sigma =  kwargs.get("streaming_update_sigma")

        self.num_updates = 0
        self.prev_num_updates = -1
        self.first_time = True

    def train_learner(self):
        eval_dict = dict()
        self.model.eval()
        for i, (images, labels, idx) in enumerate(self.train_dataloader):
            self.samples_cnt += images.size(0)
            
            acc = self.online_step(images, labels, idx)

            self.report_training(self.samples_cnt, acc)
            if self.samples_cnt > self.num_eval:
                with torch.no_grad():
                    test_sampler = OnlineTestSampler(self.test_dataset, self.exposed_classes)
                    test_dataloader = DataLoader(self.test_dataset, batch_size=self.batchsize*2, sampler=test_sampler, num_workers=self.n_worker)
                    eval_dict = self.online_evaluate(test_dataloader)
                    self.eval_results["test_acc"].append(eval_dict['avg_acc'])
                    self.eval_results["avg_acc"].append(eval_dict['cls_acc'])
                    self.eval_results["data_cnt"].append(self.samples_cnt)
                    self.report_test(self.samples_cnt, eval_dict['avg_acc'])
                    self.num_eval += self.eval_period
                sys.stdout.flush()
        if len(eval_dict)!= 0:
            self.report_test(self.samples_cnt, eval_dict['avg_acc'])

    def online_step(self, X, y, idx):
        self.add_new_class(y) 
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        X = X.to(self.device)
        y = y.to(self.device)
        batch_x_feat = self.model.features(X)
        _acc, _iter = 0.0, 0
        if self.first_time:
            for _ in range(int(self.online_iter)):   
                base_init_data = []
                base_init_labels = []          
                base_init_data.append(batch_x_feat)
                base_init_labels.append(y)
                base_init_data = torch.cat(base_init_data, dim=0)
                base_init_labels = torch.cat(base_init_labels, dim=0)
                
                self.fit_base(base_init_data, base_init_labels)
                logits = self.predict(batch_x_feat)
                _, pred_label = torch.max(logits, 1)
                pred_label = pred_label.to(self.device)
                acc = (pred_label == y).sum().item() / y.size(0)
                _acc += acc
                _iter += 1
        else:
            for _ in range(int(self.online_iter)):
                for x, label in zip(batch_x_feat, y):
                    self.fit(x.cpu(), label.view(1, ))

                logits = self.predict(batch_x_feat)
                _, pred_label = torch.max(logits, 1)
                pred_label = pred_label.to(self.device)
                acc = (pred_label == y).sum().item() / y.size(0) #  correct_cnt
                _acc += acc
                _iter += 1
        return _acc / _iter


    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)
        self.model.eval()
        with torch.no_grad():
            # initialize parameters for testing
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.n_classes))  #128,100
            mb = min(self.batchsize*2, num_samples)

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                # print('\nFirst predict since model update...computing Lambda matrix...')
                Lambda = torch.pinverse(
                    (1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.input_shape).to(
                        self.device))
                self.Lambda = Lambda
                self.prev_num_updates = self.num_updates
            else:
                Lambda = self.Lambda

            # parameters for predictions
            M = self.muK.transpose(1, 0)
            W = torch.matmul(Lambda, M)
            c = 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                self.start = min(i, num_samples - mb)
                end = i + mb
                x = X[self.start:end]
                scores[self.start:end, :] = torch.matmul(x, W) - c

            self.mask = self.mask.cpu()
            scores = scores + self.mask
            # return predictions or probabilities
            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()
            
    def fit(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        with torch.no_grad():

            # covariance updates
            if self.streaming_update_sigma:
                x_minus_mu = (x - self.muK[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Sigma = (self.num_updates * self.Sigma + delta) / (self.num_updates + 1)

            # update class means
            self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
            self.cK[y] += 1
            self.num_updates += 1


    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        
        X = X.to(self.device)
        y = y.squeeze().long()

        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)


    def online_evaluate(self, test_loader): 
        total_correct = 0.0
        total_num_data = 0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data

                x = x.to(self.device)
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                
                feat = self.model.features(x).to(self.device)
            
                logit = self.predict(feat, return_probas=False)

                pred = torch.argmax(logit, dim=-1)
                _, pred_label = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(pred_label == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()

        avg_acc = total_correct / total_num_data
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        

        eval_dict = {"avg_acc": avg_acc, "cls_acc": cls_acc}
        
        return eval_dict


    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        self.first_time = False


    def setup_model(self):
        super().setup_model()
        self.input_shape = self.model.feature_dim
        self.Sigma = torch.ones((self.input_shape, self.input_shape)).to(self.device)
        self.Lambda = torch.zeros_like(self.Sigma).to(self.device)
        self.muK = torch.zeros((self.n_classes, self.input_shape)).to(self.device)
        self.cK = torch.zeros(self.n_classes).to(self.device)
        self.model.eval()

    def report_training(self, sample_num, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_acc {train_acc:.4f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_acc):
        print(
            f"Test | Sample # {sample_num} | test_acc {avg_acc:.4f} | "
        )