import numpy as np

class NoLinearRegression:
    def __init__(self, X_train, Y_train, alpha, _lambda):
        self._lambda = _lambda
        self.X_train: np.ndarray = X_train
        self.Y_train: np.ndarray = Y_train
        self.W = [1] + list(np.random.rand(X_train.shape[1]))
        print(self.X_train.shape)
        print(len(self.W))
        self.alpha = alpha

    def H(self, X, W):
        return np.dot(X, W)
    
    def Loss(self):
        n = len(self.Y_train)
        return (1/(2*n)) * (np.linalg.norm(self.Y_train - self.H(self.X_train, self.W)) ** 2)
    
    def Lasso(self):
        return self.Loss() + (self._lambda * np.sum(np.abs(self.W)))
    
    def Ridge(self):
        return self.Loss() + (self._lambda * np.linalg.norm(self.W) ** 2)
    
    def Derivative(self):
        n = len(self.Y_train)
        return (1/n) * np.dot(self.X_train.T, self.Y_train - self.H(self.X_train, self.W))
    
    def LassoDerivative(self):
        return self.Derivative() + self._lambda * np.sum(np.abs(self.W))
    
    def RidgeDerivative(self):
        return self.Derivative() + 2 * (self._lambda * np.linalg.norm(self.W) ** 2)

    def LassoUpdate(self):
        self.W = self.W - self.alpha * self.LassoDerivative()

    def RidgeUpdate(self):
        self.W = self.W - self.alpha * self.RidgeDerivative()

    def reset(self):
        self.W = np.random.rand(self.X_train.shape[1])

    def LassoTrain(self, epochs):
        L = self.Lasso()
        L_reg = []
        for _ in range(epochs):
            self.LassoUpdate()
            self.Y_train = self.H(self.X_train, self.W)
            L = self.Lasso()
            L_reg.append(L)
        return L, L_reg

    def RidgeTrain(self, epochs):
        L = self.Ridge()
        L_reg = []
        for _ in range(epochs):
            self.RidgeUpdate()
            self.Y_train = self.H(self.X_train, self.W)
            L = self.Ridge()
            L_reg.append(L)
        return L, L_reg