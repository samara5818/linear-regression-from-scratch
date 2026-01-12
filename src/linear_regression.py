import numpy as np

class LinearRegression:
    """
    Linear Regression from scratch using Batch Gradient Descent.

    Model:
        y_hat = X w + b

    Loss (MSE):
        J(w, b) = (1/n) * sum( (y_hat - y)^2 )

    Gradients:
        dJ/dw = (2/n) * X^T (y_hat - y)
        dJ/db = (2/n) * sum( y_hat - y )
    """
        
    def __init__(self, lr=0.01, epochs=2000, fit_intercept=True, verbose=False):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.fit_intercept  = bool(fit_intercept)
        self.verbose = bool(verbose)

        self.w = None
        self.b = None
        self.loss_history = []
    
    def _to_2d(self, X):
        X= np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
    
    def fit(self, X, y):
        X = self._to_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        n, d =X.shape
        self.w = np.zeros((d, 1), dtype=float)
        self.b = 0.0
        self.loss_history = []

        for epoch in range(1, self.epochs + 1):
            # Forward pass
            y_hat = X @ self.w + (self.b if self.fit_intercept else 0.0)

            # Errors
            errors = y_hat - y

            #MSE Loss
            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)

            #Gradients
            dw = (2.0 / n) * (X.T @ errors)
            db = (2.0 / n) * np.sum(errors) if self.fit_intercept else 0.0

            #Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if self.verbose and (epoch == 1 or epoch % max(1, self.epochs // 10)==0):
                print(f"Epoch {epoch:5d}/{self.epochs} | MSE: {loss:.6f}")
        return self
    
    def predict(self, X):
        if self.w is None:
            raise RuntimeError("Model is not trained. call fit() first.")
        X = self._to_2d(X)
        y_hat = X @ self.w + (self.b if self.fit_intercept else 0.0)
        return y_hat.ravel()
    
    
    
    

