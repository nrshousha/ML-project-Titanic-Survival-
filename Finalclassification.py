

# importing sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge
from sklearn.model_selection import train_test_split

from data_loading import read_csv

class ClassificationModels:
    def __init__(self):
        # Load raw data
        header, data = read_csv("/content/train.csv")

        # Preprocess data
        self.X, self.y = preprocess_classification(data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Initialize scaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def ridge_training(self):
        model = LogisticRegression(penalty='l2', solver='saga', C=0.5, max_iter=1000, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        return model

    def sgd_training(self):
        model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.5, max_iter=1000, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        return model

# Modified part for survival prediction classification
# Preprocessing Function
def preprocess_classification(data, has_target=True):
    X, y = [], []
    for row in data:
        try:
            # Extract features
            sex = 1 if row[4].strip().lower() == 'male' else 0
            age = float(row[5]) if row[5] else 29.7  # Default age if missing
            pclass = float(row[2]) if row[2] else 3  # Default Pclass if missing
            fare = float(row[9]) if row[9] else 0    # Default fare if missing
            X.append([pclass, sex, age, fare])

            # Extract target (if present)
            if has_target:
                y.append(1 if row[1].strip().lower() == '1' else 0)  # Check if survived (1 or 0)
        except Exception as e:
            print(f"Skipping row due to error: {row}, Error: {e}")
            continue

    # Return features and targets (if applicable)
    if has_target:
        return X, y
    else:
        return X

if __name__ == "__main__":
    header, data = read_csv("train.csv")
    X, y = preprocess_classification(data)  # preprocessing

    print(f"Loaded {len(X)} samples for classification.")
    print("Sample features:", X[0])
    print("Sample target (survival):", y[0])


    model = ClassificationModels()
    logistic_model = model.ridge_training()
    print("Logistic regression model trained.")

    sgd_model = model.sgd_training()
    print("SGD model trained.")

# Logistic Regression Classifier (All Modes)
# ============================
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, mode='loss', lmbd=0.1, batch_size=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0
        self.mode = mode  # 'loss', 'ridge', 'batch', 'sgd'
        self.lmbd = lmbd
        self.batch_size = batch_size

    def sigmoid(self, z):
        return 1 / (1 + pow(2.71828, -z))

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
                    linear_model = sum([self.weights[j] * float(X[i][j]) for j in range(n_features)]) + self.bias
                    y_pred = self.sigmoid(linear_model)
                    error = y_pred - float(y[i])

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
                linear_model = sum([self.weights[j] * float(X[i][j]) for j in range(n_features)]) + self.bias
                y_pred = self.sigmoid(linear_model)
                error = y_pred - float(y[i])
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
                linear_model = sum([self.weights[j] * float(X[i][j]) for j in range(n_features)]) + self.bias
                y_pred = self.sigmoid(linear_model)
                error = y_pred - float(y[i])
                for j in range(n_features):
                    self.weights[j] -= self.learning_rate * error * float(X[i][j])
                self.bias -= self.learning_rate * error

    def predict(self, X):
        preds = []
        for x in X:
            linear_model = sum([self.weights[j] * float(x[j]) for j in range(len(x))]) + self.bias
            y_pred = self.sigmoid(linear_model)
            preds.append(1 if y_pred >= 0.5 else 0)
        return preds

# Preprocessing Functions
# ============================


def evaluate(predictions, actual):
    """
    Evaluates classification performance using accuracy, confusion matrix,
    precision, and recall.

    Parameters:
        predictions (list): Predicted binary labels (0 or 1).
        actual (list): Actual binary labels (0 or 1).

    Prints:
        Accuracy, confusion matrix (TP, FP, FN, TN), precision, and recall.
    """
    correct = sum(p == a for p, a in zip(predictions, actual))
    accuracy = correct / len(actual)

    TP = sum(p == 1 and a == 1 for p, a in zip(predictions, actual))
    TN = sum(p == 0 and a == 0 for p, a in zip(predictions, actual))
    FP = sum(p == 1 and a == 0 for p, a in zip(predictions, actual))
    FN = sum(p == 0 and a == 1 for p, a in zip(predictions, actual))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Negatives (TN): {TN}")

    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")



# Main Testing Block (All Modes)
# ============================
if __name__ == "__main__":
    header, data = read_csv("train.csv")

    Xc, yc = preprocess_classification(data)


    modes = ['loss', 'ridge', 'batch', 'sgd']

    for mode in modes:
        print(f"\n=== Logistic Regression ({mode}) ===")
        clf = LogisticRegression(learning_rate=0.01, epochs=1000, mode='loss')
        clf.train(Xc, yc)
        preds_c = clf.predict(Xc)
        print("Predictions:", [round(p, 2) for p in preds_c[:5]])
        print("Actual:     ", [round(y, 2) for y in yc[:5]])
        evaluate(preds_c, yc)
        