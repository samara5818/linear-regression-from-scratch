import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):

    X = np.asarray(X, dtype=float)
    y= np.asarray(y, dtype=float)

    n = len(y)
    idx = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    
    test_n = int(round(n*test_size))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
