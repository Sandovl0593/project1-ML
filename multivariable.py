import numpy as np

class NoLinearRegression:
    def __init__(self, X, Y, alpha, _lambda):
        self._lambda = _lambda
        self.X: np.ndarray = X
        self.Y: np.ndarray = Y
        self.W = np.random.rand(X.shape[1])
        self.alpha = alpha

    def Hipotesis(self):
        return np.dot(self.X, self.W)
    
    def Loss(self):
        n = len(self.Y)
        return (1/(2*n)) * (np.linalg.norm(self.Y - self.Hipotesis()) ** 2)
    
    def Lasso(self):
        return self.Loss() + (self._lambda * np.sum(np.abs(self.W)))
    
    def Ridge(self):
        return self.Loss() + (self._lambda * np.sum(self.W ** 2))
    
    def Derivative(self):
        n = len(self.Y)
        return (1/n) * np.dot(self.X.T, self.Hipotesis() - self.Y)
    
    def LassoDerivative(self):
        return self.Derivative() + self._lambda * np.sum(np.abs(self.W))
    
    def RidgeDerivative(self):
        return self.Derivative() + 2 * self._lambda * np.sum(self.W ** 2)
 
    def LassoUpdate(self):
        self.W = self.W - self.alpha * self.LassoDerivative()

    def RidgeUpdate(self):
        self.W = self.W - self.alpha * self.RidgeDerivative()

    def LassoTrain(self, epochs):
        for _ in range(epochs):
            self.LassoUpdate()

    def RidgeTrain(self, epochs):
        for _ in range(epochs):
            self.RidgeUpdate()