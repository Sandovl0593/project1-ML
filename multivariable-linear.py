import numpy as np

def Hipotesis(X, W) -> np.ndarray:
    return np.dot(X, W)


def Loss(X, Y: np.ndarray, W) -> np.ndarray:
    n = len(Y)
    return (1/(2*n)) * (np.linalg.norm(Y - Hipotesis(X, W)) ** 2)


def Lasso(X, Y: np.ndarray, W, _lambda):
    return Loss(X, Y, W) + (_lambda * np.sum(np.abs(W)))


def Ridge(X, Y: np.ndarray, W, _lambda):
    return Loss(X, Y, W) + (_lambda * np.sum(W ** 2))