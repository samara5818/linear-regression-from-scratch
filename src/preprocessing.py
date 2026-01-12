import numpy as np

class StandardScalarScratch:
    """
    Standard Scaler from scratch.
    Z = (X - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            X (array-like): The data used to compute the mean and std.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("The scaler has not been fitted yet. Please call 'fit' before 'transform'.")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        
        return self.fit(X).transform(X)