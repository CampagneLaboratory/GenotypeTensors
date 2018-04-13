_flatten = lambda l: [item for sublist in l for item in sublist]

class PerformanceList(list):


    def __init__(self) -> None:
        super().__init__()
        #self.list=[]

   # def append(self, values):
   #     self.list.append(values)

    def get_metric(self, metric_name):
        for pe in _flatten(self):
            metric = pe.get_metric(metric_name)
            if metric is not None:
                return metric
        return None

    def set_metric(self, iteration, metric_name, value):
        for pe in self:
            pe.observe_named_metric(iteration, metric_name, value, None, None)

    def set_metric_with_outputs(self, iteration, metric_name, loss, outputs, targets):
        for pe in self:
            pe.observe_named_metric(iteration, metric_name, loss, outputs, targets)

    def init_performance_metrics(self):
        for pe in self:
            pe.init_performance_metrics()

    def metric_names(self):
        names=[perf.metric_names() for perf in self]
        return _flatten(names)

    def estimates_of_metrics(self):
        return self.estimates_of_metric()

    def estimates_of_metric(self):
        estimates=[]
        estimates +=[perf.estimates_of_metric() for perf in self]
        return _flatten(estimates)

    def progress_message(self, metrics):
        estimators=[]
        for pe in self:
            for name in pe.metric_names():
                if name in metrics:
                    estimators+=[pe]
        return " ".join([pe.progress_message() for pe in estimators])