
from org.campagnelab.dl.performance.PerformanceEstimator import PerformanceEstimator


class FloatHelper(PerformanceEstimator):
    def __init__(self, name=""):
        self.name=name
        self.init_performance_metrics()

    def init_performance_metrics(self):

        self.value = 0
        self.count = 0

    def estimates_of_metric(self):
        if self.count==0:
            print("Invalid count for "+self.name)
            return [float('nan')]
        return [self.value/self.count]

    def metric_names(self):
        return [self.name]

    def observe_performance_metric(self, iteration, value, outputs, targets):

        self.value+=value
        self.count+=1

    def progress_message(self):
        """ Return a message suitable for logging progress of the metrics."""
        return self.name+": {:.4f}".format(*self.estimates_of_metric())