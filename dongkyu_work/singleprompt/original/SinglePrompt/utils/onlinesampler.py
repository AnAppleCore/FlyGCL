import torch
from  torch.utils.data.distributed import DistributedSampler 
from typing import Optional, Sized

class OnlineSampler(DistributedSampler):
    def __init__(self, data_source: Optional[Sized], num_tasks, m, n, rnd_seed, cur_iter = 0, varing_NM = False) -> None:

        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.generator  = torch.Generator().manual_seed(rnd_seed)
        
        self.n  = n
        self.m  = m
        self.varing_NM = varing_NM
        self.task = cur_iter

        self.disjoint_num   = len(self.classes) * n // 100
        self.disjoint_num   = int(self.disjoint_num // num_tasks) * num_tasks
        self.blurry_num     = len(self.classes) - self.disjoint_num
        # self.blurry_num     = int(self.blurry_num // num_tasks) * num_tasks

        if not self.varing_NM:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order         = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes   = class_order[:self.disjoint_num]
            self.disjoint_classes   = self.disjoint_classes.reshape(num_tasks, -1).tolist()
            self.blurry_classes     = class_order[self.disjoint_num:self.disjoint_num + self.blurry_num]
            self.blurry_classes     = self.blurry_classes.reshape(num_tasks, -1).tolist()

            print("disjoint classes {}: ".format(self.disjoint_num ), self.disjoint_classes)
            print("blurry classes {}: ".format(self.blurry_num ), self.blurry_classes)
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            for i in range(num_tasks):
                blurred += self.blurry_indices[i][:len(self.blurry_indices[i]) * m // 100]
                self.blurry_indices[i] = self.blurry_indices[i][len(self.blurry_indices[i]) * m // 100:]
            blurred = torch.tensor(blurred)
            blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
            print("blurry indices: ", len(blurred))
            num_blurred = len(blurred) // num_tasks
            for i in range(num_tasks):
                self.blurry_indices[i] += blurred[:num_blurred]
                blurred = blurred[num_blurred:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()
        else:
            # Divide classes into N% of disjoint and (100 - N)% of blurry
            class_order = torch.randperm(len(self.classes), generator=self.generator)
            self.disjoint_classes = class_order[:self.disjoint_num].tolist()
            if self.disjoint_num > 0:
                if n > 100:
                    num_cls_per_task = int(len(self.classes) / num_tasks)
                    self.disjoint_classes = [[i for i in range(j*num_cls_per_task, (j+1)*num_cls_per_task)] for j in range(num_tasks)]
                else:
                    self.disjoint_slice = [0] + torch.randint(0, self.disjoint_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.disjoint_num]
                    self.disjoint_classes = [self.disjoint_classes[self.disjoint_slice[i]:self.disjoint_slice[i + 1]] for i in range(num_tasks)]
            else:
                self.disjoint_classes = [[] for _ in range(num_tasks)]

            if self.blurry_num > 0:
                if m > 100:
                    # assert len(self.classes) == num_tasks
                    self.blurry_classes = [[i] for i in range(len(self.classes))]
                else:
                    self.blurry_slice = [0] + torch.randint(0, self.blurry_num, (num_tasks - 1,), generator=self.generator).sort().values.tolist() + [self.blurry_num]
                    self.blurry_classes = [class_order[self.disjoint_num + self.blurry_slice[i]:self.disjoint_num + self.blurry_slice[i + 1]].tolist() for i in range(num_tasks)]
            else:
                self.blurry_classes = [[] for _ in range(num_tasks)]

            print("disjoint classes: ", self.disjoint_classes)
            print("blurry classes: ", self.blurry_classes)
            
            # Get indices of disjoint and blurry classes
            self.disjoint_indices   = [[] for _ in range(num_tasks)]
            self.blurry_indices     = [[] for _ in range(num_tasks)]
            num_blurred = 0
            for i in range(len(self.targets)):
                for j in range(num_tasks):
                    if self.targets[i] in self.disjoint_classes[j]:
                        self.disjoint_indices[j].append(i)
                        break
                    elif self.targets[i] in self.blurry_classes[j]:
                        self.blurry_indices[j].append(i)
                        num_blurred += 1
                        break

            # Randomly shuffle M% of blurry indices
            blurred = []
            num_blurred = num_blurred * m // 100

            if m > 100:
                blurred_101 = []
                # Split each class into subset: class 1, 100 subset
                for i in range(num_tasks):
                    num_subset = num_tasks - i
                    num_samples = len(self.blurry_indices[i])
                    ratio = 0.75
                    start = int(num_samples*ratio)
                    temp = torch.randint(start, num_samples, (num_subset-1,), generator=self.generator).sort().values.tolist()
                    split_idx = [0] + temp + [num_samples]
                    split = [self.blurry_indices[i][split_idx[j]:split_idx[j + 1]] for j in range(num_subset)]
                    blurred_101.append(split)
                    
                for i in range(num_tasks):
                    task_indices = []
                    for j in range(0, i+1):
                        task_indices += blurred_101[j][i-j]
                    self.blurry_indices[i] = task_indices
                

            elif num_blurred > 0:
                num_blurred = [0] + torch.randint(0, num_blurred, (num_tasks-1,), generator=self.generator).sort().values.tolist() + [num_blurred]

                for i in range(num_tasks):
                    blurred += self.blurry_indices[i][:num_blurred[i + 1] - num_blurred[i]]
                    self.blurry_indices[i] = self.blurry_indices[i][num_blurred[i + 1] - num_blurred[i]:]
                blurred = torch.tensor(blurred)
                blurred = blurred[torch.randperm(len(blurred), generator=self.generator)].tolist()
                print("blurry indices: ", len(blurred))
                # num_blurred = len(blurred) // num_tasks
                for i in range(num_tasks):
                    self.blurry_indices[i] += blurred[:num_blurred[i + 1] - num_blurred[i]]
                    blurred = blurred[num_blurred[i + 1] - num_blurred[i]:]
            
            self.indices = [[] for _ in range(num_tasks)]
            for i in range(num_tasks):
                print("task %d: disjoint %d, blurry %d" % (i, len(self.disjoint_indices[i]), len(self.blurry_indices[i])))
                # print(self.blurry_indices[i])
                self.indices[i] = self.disjoint_indices[i] + self.blurry_indices[i]
                self.indices[i] = torch.tensor(self.indices[i])[torch.randperm(len(self.indices[i]), generator=self.generator)].tolist()

        self.num_samples = int(len(self.indices[self.task]))
        self.total_size = self.num_samples
        self.num_selected_samples = int(len(self.indices[self.task]))

    def __iter__(self):
        return iter(self.indices[self.task])

    def __len__(self):
        return self.num_selected_samples

    def set_task(self, cur_iter):

        if cur_iter >= len(self.indices) or cur_iter < 0:
            raise ValueError("task out of range")
        self.task = cur_iter

        self.num_samples = int(len(self.indices[self.task]))
        self.total_size = self.num_samples
        self.num_selected_samples = int(len(self.indices[self.task]))


class OnlineTestSampler(DistributedSampler):
    def __init__(self, data_source: Optional[Sized], exposed_class) -> None:
        self.data_source    = data_source
        self.classes    = self.data_source.classes
        self.targets    = self.data_source.targets
        self.exposed_class  = exposed_class
        self.indices    = [i for i in range(self.data_source.__len__()) if self.targets[i] in self.exposed_class]

        self.num_samples = int(len(self.indices))
        self.total_size = self.num_samples
        self.num_selected_samples = int(len(self.indices))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_selected_samples