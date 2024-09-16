import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class NonLinearRegresion:
    def __init__(self, grado, X, Y, alpha, _lambda):
        np.random.seed(2005)
        self.grado = grado
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self._lambda = _lambda
        self.W = np.random.rand(grado+1)

    def H(self, X):
        return np.dot(X, self.W)

    def Loss(self):
        n = len(self.Y)
        y_pred = self.H(self.X)
        return (1/(2*n)) * (np.linalg.norm(self.Y - y_pred) ** 2)

    def Loss_Lasso(self):
        return self.Loss() + (self._lambda * np.linalg.norm(self.W, 1))

    def Loss_Ridge(self):
        return self.Loss() + (self._lambda * np.linalg.norm(self.W, 2) ** 2)

    def Derivative(self):
        n = len(self.Y)
        return (-1/n) * np.matmul(self.X.T, self.Y - self.H(self.X))

    def Derivative_Lasso(self):
        return self.Derivative() + (self._lambda * np.sign(self.W))

    def Derivative_Ridge(self):
        return self.Derivative() + (2 * self._lambda * self.W)

    def Update_Lasso(self):
        self.W -= self.alpha * self.Derivative_Lasso()

    def Update_Ridge(self):
        self.W -= self.alpha * self.Derivative_Ridge()

    def predic(self, x):
        potencias = np.arange(self.grado + 1)
        x_poly = np.vstack([x**i for i in range(self.grado + 1)]).T
        return self.H(x_poly, self.W)

    def Train_Lasso(self, epochs):
        L_reg = []

        potencias = np.arange(self.grado + 1)
        self.X = np.power.outer(self.X, potencias)

        for _ in range(epochs):
            self.Update_Lasso()
            L = self.Loss_Lasso()
            L_reg.append(L)
        return L_reg

    def Train_Ridge(self, epochs):
        L_reg = []

        for _ in range(epochs):
            self.Update_Ridge()
            L = self.Loss_Ridge()
            L_reg.append(L)
        return L_reg


# class NonLinearRegresion:
#     def __init__(self, grado):
#         np.random.seed(2005)
#         self.m_W = np.random.rand(grado+1)
#         self.m_b = np.random.random()
#         self.grado = grado

#     def H(self, X):
#         return np.dot(X, self.m_W)

#     def predic(self, x):
#         potencias = np.arange(self.grado + 1)
#         x = np.power.outer(x, potencias)
#         return np.dot(x, self.m_W)

#     def Loss_L2(self, X, Y, lambda_):
#         y_pred = self.H(X)
#         return (np.linalg.norm((Y - y_pred))**2)/(2*len(Y)), y_pred + lambda_*np.linalg.norm(self.m_W, 2)

#     def Loss_L1(self, X, Y, lambda_):
#         y_pred = self.H(X)
#         return (np.linalg.norm((Y - y_pred))**2) / (2 * len(Y)), y_pred + lambda_ * np.linalg.norm(self.m_W, 1)

#     def dL(self, X, Y, Y_pre, lambda_):
#         dw = np.matmul(Y - Y_pre, -X)/len(Y) + 2*lambda_*self.m_W
#         # db = np.sum((Y - Y_pre)*(-1))/len(Y)
#         return dw

#     def change_params(self, dw, alpha):
#         self.m_W = self.m_W - alpha*dw
#         # self.m_b = self.m_b - alpha*db

#     def train(self, X, Y, alpha, epochs, lambda_, reg):
#         error_list = []
#         time_stamp = []

#         potencias = np.arange(self.grado + 1)
#         X = np.power.outer(X, potencias)

#         for i in range(epochs):
#             if reg == "L2":
#                 loss, y_pred = self.Loss_L2(X, Y, lambda_)
#             elif reg == "L1":
#                 loss, y_pred = self.Loss_L1(X, Y, lambda_)
#             time_stamp.append(i)
#             error_list.append(loss)
#             dw = self.dL(X, Y, y_pred, lambda_)
#             self.change_params(dw, alpha)

#             if (i % 1000 == 0):
#                 # self.plot_error(time_stamp, error_list)
#                 print("error de pérdida : " + str(loss))
#                 # LR.plot_line(x, LR.predic(x))
#         return time_stamp, error_list

#     def plot_error(self, time, loss, reg_type):
#         color = 'red' if reg_type == 'L1' else 'blue'
#         sns.lineplot(x=time, y=loss, color=color)
#         plt.title(f'Error de entrenamiento - Grado {self.grado} - {reg_type}')
#         plt.xlabel('Epochs')
#         plt.savefig(f"Error_Grado_{self.grado}_{reg_type}.png")
#         plt.close()

#     def plot_line(self, x, y_pre,  reg_type, etiquetax='META', etiquetay='MTO_PIA'):
#         color = 'red' if reg_type == 'L1' else 'blue'
#         sns.lineplot(x=x, y=y_pre, label=f'Grado {
#                      self.grado} - {reg_type}', color=color)
#         plt.xlabel(etiquetax)
#         plt.ylabel(etiquetay)
#         plt.title(f'Línea de Regresión - Grado {self.grado} - {reg_type}')
#         plt.legend()
#         plt.savefig(f"Prediccion_Grado_{self.grado}_{reg_type}.png")
#         plt.close()


df = pd.read_csv("forestfires.csv")
Y = df["area"].to_numpy()
X = df.drop(columns=["month", "day", "area"]).to_numpy()

# # Convertir en valores 

# df = df.drop(columns=["month", "day", "area"])

# # OBTENER LA CORRELACION DE LOS POSIBLES CANDIDATOS DE X RESPECTO A Y PARA SABER CUAL O CUALES SON LOS MEJORES CANDIDATOS
# for feature in df.columns:
#     x = df[feature].to_numpy()
#     corr = np.corrcoef(x, y)
#     print('Correlación entre', feature, 'y area:', corr[0, 1]**2)


# # GRAFICAR LA CORRELACION DE LOS POSIBLES CANDIDATOS DE X RESPECTO A Y
# for feature in df.columns:
#     x = df[feature].to_numpy()
#     plt.title(feature + ' vs area')
#     plt.scatter(x,y)
#     plt.xlabel(feature)
#     plt.ylabel("area")
#     plt.show()

print(X.shape)
print(Y.shape)

grado = 2        
alpha = 0.01     
_lambda = 0.1    

modelo_lasso = NonLinearRegresion(grado, X, Y, alpha, _lambda)

# modelo_ridge = NonLinearRegresion(grado, X, Y, alpha, _lambda)

epochs = 1000

historial_lasso = modelo_lasso.Train_Lasso(epochs)

# historial_ridge = modelo_ridge.Train_Ridge(epochs)

pérdida_lasso = modelo_lasso.Loss_Lasso()
print(f"Pérdida Lasso: {pérdida_lasso}")

# pérdida_ridge = modelo_ridge.Loss_Ridge()
# print(f"Pérdida Ridge: {pérdida_ridge}")

# predicciones_lasso = modelo_lasso.predic(X)
# predicciones_ridge = modelo_ridge.predic(X)