# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> optim
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   08/09/2021 12:06
=================================================='''
from torch.optim.lr_scheduler import _LRScheduler, StepLR


# class PolyLR(_LRScheduler):
#     def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
#         self.power = power
#         self.max_iters = max_iters  # avoid zero lr
#         self.min_lr = min_lr
#         super(PolyLR, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)]


class PolyLR(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr