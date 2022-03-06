from collections import Counter
import numpy as np

def euclidean_dist(x1,x2):
    return np.sqrt(np.sum( (x1-x2) ** 2 ))

class kNN:

    def __init__(self,k_val=3) -> None:
        self.k = k_val

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred =  [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self,x):
        dist = [euclidean_dist(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(dist)[:self.k]
        k_near_label = [self.y_train[i] for i in k_idx]
        most_common_val = Counter(k_near_label).most_common(1)
        return most_common_val[0][0]


if __name__ == "__main__":

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from mlutils import accuracy

    iris = datasets.load_iris()
    X,y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

    clf = kNN(k_val=3)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print("accuracy==>", accuracy(y_test, preds))
    