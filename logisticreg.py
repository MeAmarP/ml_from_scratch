import numpy as np



class LogisticRegression:

    def __init__(self, lr=0.001, n_iter=500):
        self.learning_rate = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None


    def fit(self,X, y):

        #Get N and # of features
        n_samples, n_features = X.shape

        # Init weights and biases
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iter):
            linear_val = np.dot(X,self.weights) + self.bias
            y_pred = self._sigmoid(linear_val)

            # Calculate Gradient of weights and biases. 
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # update rule for weight and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, X):
        linear_val = np.dot(X,self.weights) + self.bias
        y_pred = self._sigmoid(linear_val)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls)

    def _sigmoid(self, val):
        return 1 / (1 + np.exp(-val))

if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from mlutils import accuracy
    
    bc_dataset = datasets.load_breast_cancer()
    X,y = bc_dataset.data, bc_dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

    log_regressor = LogisticRegression(lr=0.0001, n_iter=1500)

    log_regressor.fit(X_train, y_train)

    preds = log_regressor.predict(X_test)

    print("Accuracy--->", accuracy(y_test, preds))


    

    # import matplotlib.pyplot as plt
    # # cmap = plt.get_cmap("terrain")
    # fig = plt.figure(figsize=(8, 6))
    # m1 = plt.scatter(X_train, y_train, s=10)
    # m2 = plt.scatter(X_test, y_test, s=10)
    # plt.plot(X_test, preds, color="black", linewidth=2, label="Prediction")
    # plt.show()


 