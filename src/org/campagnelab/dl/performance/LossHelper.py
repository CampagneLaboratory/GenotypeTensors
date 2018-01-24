import torch
from org.campagnelab.dl.performance.PerformanceEstimator import PerformanceEstimator


class LossHelper(PerformanceEstimator):
    """
        Keep estimates of the average loss.
    """
    def __init__(self, loss_name="loss"):
        self.loss_name=loss_name
        self.init_performance_metrics()

    def init_performance_metrics(self):
        self.max_iteration=0
        self.loss_acc = 0

    def estimates_of_metric(self):
        average_loss=self.loss_acc/(self.max_iteration+1)
        return [average_loss]

    def metric_names(self):
        return [self.loss_name]

    def observe_performance_metric(self, iteration, loss, outputs, targets):
        self.max_iteration = iteration
        self.loss_acc+=loss

    def progress_message(self):
        """ Return a message suitable for logging progress of the metrics."""
        return "{}: {:.4f}".format(self.loss_name, self.estimates_of_metric()[0])