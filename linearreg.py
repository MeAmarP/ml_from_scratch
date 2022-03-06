from unittest import skipUnless
import numpy as np



class LinearRegression:

    def __init__(self, lr=0.001, n_iter=500):
        self.learning_rate = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None


    def fit(self,X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X,self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):
        y_approx = np.dot(X, self.weights) + self.bias
        return y_approx

if __name__ == "__main__":
 from sklearn import datasets
 from sklearn.model_selection import train_test_split


 