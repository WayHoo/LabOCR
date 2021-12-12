import math
from paddle.optimizer.lr import LRScheduler


class CyclicalCosineDecay(LRScheduler):
    def __init__(self,
                 learning_rate,
                 T_max,
                 cycle=1,
                 last_epoch=-1,
                 eta_min=0.0,
                 verbose=False):
        """
        Cyclical cosine learning rate decay
        A learning rate which can be referred in https://arxiv.org/pdf/2012.12645.pdf
        Args:
            learning rate(float): learning rate
            T_max(int): maximum epoch num
            cycle(int): period of the cosine decay
            last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
            eta_min(float): minimum learning rate during training
            verbose(bool): whether to print learning rate for each epoch
        """
        super(CyclicalCosineDecay, self).__init__(learning_rate, last_epoch,
                                                  verbose)
        self.cycle = cycle
        self.eta_min = eta_min

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        reletive_epoch = self.last_epoch % self.cycle
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                (1 + math.cos(math.pi * reletive_epoch / self.cycle))
        return lr
