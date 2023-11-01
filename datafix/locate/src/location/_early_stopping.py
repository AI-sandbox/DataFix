################################################################################
# EarlyStopping class.
################################################################################


class EarlyStopping:
    """
    Early stops the algorithm if the metric does not improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False):
        """
        Parameters
        ----------
        patience : int, default=10
            Maximum number of iterations without improvement in metric.
        verbose : bool, default=False
            Print messages.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_metric = 0
        self.verbose = verbose

    def __call__(self, metric: float):
        """
        Evaluate the early stopping and update the class parameters.

        Parameters
        ----------
        metric : float
            Metric to consider for stopping criteria.

        Returns:
            (None)

        """
        score = metric

        # First iteration
        if self.best_score is None:
            self.best_score = score

        # Best iteration
        elif score < self.best_score or score >= 0.8:
            self.best_score = score
            self.counter = 0

        # Worse iteration
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
