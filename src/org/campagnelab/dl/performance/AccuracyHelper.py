import torch

from org.campagnelab.dl.performance.PerformanceEstimator import PerformanceEstimator


class AccuracyHelper(PerformanceEstimator):
    def __init__(self, prefix=""):
        self.prefix=prefix
        self.init_performance_metrics()
        self.device = torch.device("cpu")

    def init_performance_metrics(self):

        self.total = 0
        self.correct = 0
        self.accuracy = 0

    def estimates_of_metric(self):
        if self.total==0: return [float('nan'),self.correct,self.total]
        accuracy = 100. * self.correct / self.total
        return [accuracy, self.correct, self.total]

    def metric_names(self):
        return [self.prefix+"accuracy", self.prefix+"correct", self.prefix+"total"]

    def observe_performance_metric(self, iteration, loss, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets.data).to(self.device).sum().item()

    def progress_message(self):
        """ Return a message suitable for logging progress of the metrics."""
        return "acc: {:.4f} {}/{}".format(*self.estimates_of_metric())