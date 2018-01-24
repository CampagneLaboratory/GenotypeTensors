class PerformanceEstimator:
    def init_performance_metrics(self):
        """Initializes accumulators. Must be called before starting another round of estimations.
        """
        pass

    def observe_named_metric(self, iteration, metric_name, loss, outputs, targets):
        if metric_name in self.metric_names():
            self.observe_performance_metric( iteration, loss, outputs, targets)

    def observe_performance_metric(self, iteration, loss, outputs, targets):
        """Collect information to calculate metrics. Must be called at each iteration (mini-batch).
        """
        pass


    def __iter__(self):
        """Support the iterable interface by wrapping itself in a list. """
        return iter([ self ])

    def metric_names(self):
        """
        Names of estimated metrics.
        :return: list of names of evaluated metrics.
        """
        return []

    def estimates_of_metric(self):
        """
        Return a list of metric estimates.
        :return: List of float values for evaluated metrics.
        """
        return []

    def progress_message(self):
        """ Return a message suitable for logging progress of the metrics."""
        return ""

    def get_metric(self, metric_name):
        """ return the metric estimate corresponding to this name, or None if not estimated
        by this estimator."""
        for index in range(len(self.metric_names())):
            if metric_name in self.metric_names():
                return self.estimates_of_metric()[index]
        return None
