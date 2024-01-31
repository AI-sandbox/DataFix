class SupervisedImputer:
    def __init__(self, estimator, n_corrupted):
        self.estimator = estimator
        self.n_corrupted = n_corrupted

    def fit(self, x):
        self.estimator.fit(x[:, self.n_corrupted :], x[:, 0 : self.n_corrupted])
        return self

    def transform(self, x):
        x = x.copy()
        y = self.estimator.predict(x[:, self.n_corrupted :])
        x[:, 0 : self.n_corrupted] = y
        return x
