# learning rate schedules that help train ureg regularized models.
# The trick is to reset the learning rate to a high value periodically (learning rate annealing) to allow the optimizer
# to explore the shaved loss space.
import sys
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, LambdaLR, ReduceLROnPlateau


def construct_scheduler(optimizer, direction='min',
                        lr_patience=1,
                        extra_patience=0, ureg_reset_every_n_epoch=None,
                        factor=0.5, min_lr=1E-9):
    delegate_scheduler = ReduceLROnPlateau(optimizer, direction, factor=factor, min_lr=min_lr,
                                           patience=lr_patience + extra_patience,
                                           verbose=True)

    if ureg_reset_every_n_epoch is None:
        scheduler = delegate_scheduler
    else:
        scheduler = LearningRateAnnealing(optimizer,
                                          anneal_every_n_epoch=ureg_reset_every_n_epoch,
                                          delegate=delegate_scheduler)
    return scheduler


class LearningRateAnnealing(ReduceLROnPlateau):
    def __init__(self, optimizer, anneal_every_n_epoch=None, delegate=None):
        assert anneal_every_n_epoch is not None, "anneal_every_n_epoch must be defined"
        self.current_epoch = 0
        self.anneal_every_n_epoch = anneal_every_n_epoch
        if delegate is None:
            delegate = ExponentialLR(optimizer, gamma=0.5)
        # the     resetLR will reset the learning rate to its initial value on each step we call it:
        self.resetLR = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
        self.delegate = delegate
        self.optimizer = optimizer
        super().__init__(optimizer)

    def step(self, metrics, epoch=None):
        self.current_epoch = self.current_epoch + 1
        if (self.current_epoch > self.anneal_every_n_epoch):
            # we have reached the number of epoch to anneal:
            self.resetLR.step(epoch)
            self.current_epoch = 0
        else:
            # simply decay according to the other delegate:
            self.delegate.step(metrics, epoch)

    def lr(self):
        """Get the current learning rate for this schedule/optimizer.
         :return a tuple with the min and max learning rates for this optimizer. """
        max_lr = -1
        min_lr = sys.maxsize
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            max_lr = max(old_lr, max_lr)
            min_lr = min(old_lr, min_lr)

        return (min_lr, max_lr)
