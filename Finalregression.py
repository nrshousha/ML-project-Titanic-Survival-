from sklearn.linear_model import LinearRegression as SklearnLinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import csv
from data_loading import read_csv
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, mode='loss', lmbd=0.1, batch_size=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0
        self.mode = mode
        self.lmbd = lmbd
        self.batch_size = batch_size

    def train(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        if self.mode == 'batch':
            self.batch_gradient_descent(X, y)
        elif self.mode == 'sgd':
            self.stochastic_gradient_descent(X, y)
        else:
            for _ in range(self.epochs):
                for i in range(n_samples):
                    prediction = sum([self.weights[j] * float(X[i][j]) for j in range(n_features)]) + self.bias
                    error = prediction - float(y[i])
                    for j in range(n_features):
                        reg_term = self.lmbd * self.weights[j] if self.mode == 'ridge' else 0
                        self.weights[j] -= self.learning_rate * (error * float(X[i][j]) + reg_term)
                    self.bias -= self.learning_rate * error

    def batch_gradient_descent(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        for _ in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                prediction = sum([self.weights[j] * float(X[i][j]) for j in range(n_features)]) + self.bias
                error = prediction - float(y[i])
                for j in range(n_features):
                    dw[j] += error * float(X[i][j])
                db += error
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * (dw[j] / n_samples)
            self.bias -= self.learning_rate * (db / n_samples)

    def stochastic_gradient_descent(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        for _ in range(self.epochs):
            for i in range(n_samples):
                prediction = sum([self.weights[j] * float(X[i][j]) for j in range(n_features)]) + self.bias
                error = prediction - float(y[i])
                for j in range(n_features):
                    self.weights[j] -= self.learning_rate * error * float(X[i][j])
                self.bias -= self.learning_rate * error

    def predict(self, X):
        preds = []
        for x in X:
            prediction = sum([self.weights[j] * float(x[j]) for j in range(len(x))]) + self.bias
            preds.append(prediction)
        return preds
# Mean Absolute Error for regression

def mean_absolute_error(predictions, actual):
    total_error = sum([abs(p - float(a)) for p, a in zip(predictions, actual)])
    return total_error / len(actual)

# Preprocessing Function

def preprocess_regression(data):
        X, y = [], []
        for row in data:
            try:
                sex = 1 if row[4].strip().lower() == 'male' else 0
                age = float(row[5]) if row[5] else 29.7
                pclass = float(row[2])
                sibsp = float(row[6])
                parch = float(row[7])
                fare = float(row[9])
                X.append([pclass, sex, age, sibsp, parch])
                y.append(fare)
            except:
                continue
        return X, y

# Example usage
if __name__ == "__main__":
    header, data = read_csv("/content/train.csv")

    Xr, yr = preprocess_regression(data)

    modes = ['loss', 'ridge', 'batch', 'sgd']

    for mode in modes:
        print(f"\n=== Linear Regression ({mode}) ===")
        reg = LinearRegression(learning_rate=0.0001, epochs=1000, mode=mode)
        reg.train(Xr, yr)
        preds_r = reg.predict(Xr)
        print("Predictions:", [round(p, 2) for p in preds_r[:5]])
        print("Actual:     ", [round(y, 2) for y in yr[:5]])
        print(f"Mean Absolute Error: {mean_absolute_error(preds_r, yr):.2f}")


# ###########################


# Preprocessing Function
def preprocess_regression(data):
    X, y = [], []
    for row in data:
        try:
            sex = 1 if row[4].strip().lower() == 'male' else 0
            age = float(row[5]) if row[5] else 29.7
            pclass = float(row[2])
            sibsp = float(row[6])
            parch = float(row[7])
            fare = float(row[9])
            X.append([pclass, sex, age, sibsp, parch])
            y.append(fare)
        except:
            continue
    return X, y

# CSV reader
# def read_csv(filename):
#     with open(filename, newline='', encoding='utf-8') as f:
#         reader = csv.reader(f)
#         data = list(reader)
#         return data[0], data[1:]

# Example usage
if __name__ == "__main__":
    header, data = read_csv("/Users/Noura/Desktop/work/college/level 002/second sem/ML/Titanic project/data set/train.csv")
    Xr, yr = preprocess_regression(data)

    scaler = StandardScaler()
    Xr_scaled = scaler.fit_transform(Xr)

    modes = ['loss', 'ridge', 'batch', 'sgd']

    for mode in modes:
        print(f"\n=== Linear Regression ({mode}) ===")

        if mode == 'ridge':
            model = Ridge(alpha=0.1, solver='auto', max_iter=1000)
        elif mode == 'batch':
            model = SklearnLinearRegression()
        elif mode == 'sgd':
            model = SGDRegressor(learning_rate='constant', eta0=0.0001, max_iter=1000)
        else:  # default mode 'loss'
            model = SklearnLinearRegression()

        model.fit(Xr_scaled, yr)
        preds_r = model.predict(Xr_scaled)

        print("Predictions:", [round(p, 2) for p in preds_r[:5]])
        print("Actual:     ", [round(y, 2) for y in yr[:5]])
        print(f"Mean Absolute Error: {mean_absolute_error(preds_r, yr):.2f}")