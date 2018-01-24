import sys

from org.campagnelab.dl.performance.PerformanceEstimator import PerformanceEstimator


class LearningRateHelper(PerformanceEstimator):
    """
        Record the range of learning rates used by a scheduler/optimizer.
    """

    def __init__(self, scheduler, learning_rate_name="lr", initial_learning_rate=None):
        self.learning_rate_name = learning_rate_name
        self.init_performance_metrics()
        self.scheduler = scheduler
        self.initial_learning_rate = initial_learning_rate

    def init_performance_metrics(self):
        self.max_iteration = 0
        self.max_lr = 0
        self.min_lr = 0

    def estimates_of_metric(self):
        if self.scheduler is None:
            if self.initial_learning_rate is None:
                return [float('nan')]
            else:
                return [self.initial_learning_rate]
        min_lr, max_lr = self.lr(self.scheduler.optimizer)
        if min_lr == max_lr:
            return ["{:.2e}".format(min_lr)]
        else:
            return ["[{:.2e}.{:.2e}]".format(min_lr, max_lr)]

    def metric_names(self):
        return ["range_" + self.learning_rate_name, ]

    def observe_performance_metric(self, iteration, loss, outputs, targets):
        # nothing to observe
        pass

    def progress_message(self):
        """ Return a message suitable for logging progress of the metrics."""
        return "{}: [{:e}-{:e}]".format(self.learning_rate_name, *self.estimates_of_metric())

    def lr(self, optimizer):
        """Get the current learning rate for this schedule/optimizer.
         :return a tuple with the min and max learning rates for this optimizer. """
        max_lr = -1
        min_lr = sys.maxsize
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
            max_lr = max(old_lr, max_lr)
            min_lr = min(old_lr, min_lr)

        return (min_lr, max_lr)
