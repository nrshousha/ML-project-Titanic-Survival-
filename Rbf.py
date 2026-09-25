import math

# RBF Kernel Function
def rbf_kernel(x1, x2, gamma=1.0):
    squared_distance = sum((float(a) - float(b)) ** 2 for a, b in zip(x1, x2))
    return math.exp(-gamma * squared_distance)

# حساب مصفوفة الكيرنل
def compute_kernel_matrix(X, kernel_func, gamma=1.0):
    n_samples = len(X)
    K = [[0.0 for _ in range(n_samples)] for _ in range(n_samples)]
    for i in range(n_samples):
        for j in range(n_samples):
            K[i][j] = kernel_func(X[i], X[j], gamma)
    return K

# Kernelized Logistic Regression
class KernelLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, gamma=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gamma = gamma
        self.alpha = []
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        n_samples = len(X)
        self.alpha = [0.0] * n_samples
        self.bias = 0.0
        K = compute_kernel_matrix(X, rbf_kernel, self.gamma)
        for _ in range(self.epochs):
            for i in range(n_samples):
                linear_sum = sum(self.alpha[j] * K[j][i] for j in range(n_samples)) + self.bias
                y_pred = self.sigmoid(linear_sum)
                error = y_pred - y[i]
                for j in range(n_samples):
                    self.alpha[j] -= self.learning_rate * error * K[j][i]
                self.bias -= self.learning_rate * error

    def predict(self, X_train, X_test):
        preds = []
        for x in X_test:
            linear_sum = sum(self.alpha[j] * rbf_kernel(X_train[j], x, self.gamma) for j in range(len(X_train))) + self.bias
            y_pred = self.sigmoid(linear_sum)
            preds.append(1 if y_pred >= 0.5 else 0)
        return preds

# مثال على الاستخدام
if __name__ == "__main__":
    from data_loading import read_csv
    from Finalclassification import preprocess_classification

    header, data = read_csv("train.csv")
    X, y = preprocess_classification(data)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = KernelLogisticRegression(learning_rate=0.01, epochs=300, gamma=0.1)
    model.fit(X_train, y_train)

    preds = model.predict(X_train, X_test)

    def evaluate(predictions, actual):
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

    evaluate(preds, y_test)