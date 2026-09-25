import math
from data_loading import read_csv
from Finalclassification import preprocess_classification

# دالة Polynomial Kernel
def polynomial_kernel(x1, x2, gamma=1.0, c=1.0, degree=3):
    dot = sum(float(a) * float(b) for a, b in zip(x1, x2))
    return (gamma * dot + c) ** degree

# حساب مصفوفة الكيرنل
def compute_kernel_matrix(X, kernel_func, **kernel_params):
    n_samples = len(X)
    K = [[0.0 for _ in range(n_samples)] for _ in range(n_samples)]
    for i in range(n_samples):
        for j in range(n_samples):
            K[i][j] = kernel_func(X[i], X[j], **kernel_params)
    return K

# نموذج Kernel Logistic Regression
class KernelLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, gamma=1.0, c=1.0, degree=3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gamma = gamma
        self.c = c
        self.degree = degree
        self.alpha = []
        self.bias = 0.0

    def sigmoid(self, z):
        # حماية من overflow
        if z < -100: return 0.0
        if z > 100: return 1.0
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        n_samples = len(X)
        self.alpha = [0.0] * n_samples
        self.bias = 0.0
        # حساب مصفوفة الكيرنل باستخدام polynomial kernel
        K = compute_kernel_matrix(
            X,
            polynomial_kernel,
            gamma=self.gamma,
            c=self.c,
            degree=self.degree
        )
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
            linear_sum = sum(
                self.alpha[j] * polynomial_kernel(X_train[j], x, gamma=self.gamma, c=self.c, degree=self.degree)
                for j in range(len(X_train))
            ) + self.bias
            y_pred = self.sigmoid(linear_sum)
            preds.append(1 if y_pred >= 0.5 else 0)
        return preds

# دالة التقييم
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

# مثال كامل على الاستخدام
if __name__ == "__main__":
    # تحميل البيانات
    header, data = read_csv("train.csv")
    X, y = preprocess_classification(data)

    # تقسيم البيانات (80% تدريب، 20% اختبار)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # تدريب النموذج
    model = KernelLogisticRegression(learning_rate=0.01, epochs=300, gamma=1.0, c=1.0, degree=3)
    model.fit(X_train, y_train)

    # التنبؤ على بيانات الاختبار
    preds = model.predict(X_train, X_test)

    # تقييم النتائج
    evaluate(preds, y_test)