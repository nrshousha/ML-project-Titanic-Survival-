from data_loading import read_csv
import math

def preprocess_data(header, data):
    """Preprocesses the raw data into features and labels."""
    X = []
    y = []
    categorical_features = ['Sex', 'Embarked']
    numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    
    # Encode categorical features
    unique_values = {}
    for feature in categorical_features:
        unique_values[feature] = sorted(set(row[header.index(feature)] for row in data if row[header.index(feature)] != ''))

    # computing median for numerical data
    median_values = {}
    for feature in numerical_features:
        values = [float(row[header.index(feature)]) for row in data if row[header.index(feature)] != '']
        median = sorted(values)[len(values) // 2] if values else 0
        median_values[feature] = median
    # Fill missing values and construct feature matrix
    for row in data:
        if row[header.index('Survived')] == '' :
            continue  # Skip rows with missing critical data
        
        # Target variable
        y.append(int(row[header.index('Survived')]))
        
        # Features
        x_row = []
        for feature in numerical_features:
            value = row[header.index(feature)]
            if value == '':
                x_row.append(median_values[feature])
            else:
                x_row.append(float(value))
        
        for feature in categorical_features:
            categories = unique_values[feature]
            one_hot_vector = [1 if row[header.index(feature)] == category else 0 for category in categories]
            x_row.extend(one_hot_vector)
        
        X.append(x_row)
    
    return X, y

def scale_features(X,n_numerical_features):
    """Scales numerical features"""
    means = [sum(x[i] for x in X) / len(X) for i in range(n_numerical_features)]
    stds = [math.sqrt(sum((x[i] - means[i]) ** 2 for x in X) / len(X)) for i in range(n_numerical_features)]

    scaled_X = []
    for x in X:
        scaled_row = [(x[i] - means[i]) / stds[i] if stds[i] != 0 else 0 for i in range(n_numerical_features)]
        scaled_row.extend(x[n_numerical_features:])  # Append non-numerical features without scaling
        scaled_X.append(scaled_row)
    return scaled_X

def dot_product(a, b):
    """Computes the dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))

def matrix_multiply(A, B):
    """Multiplies two matrices A and B."""
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def transpose(matrix):
    """Transposes a matrix."""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def solve_qp(P, q, G, h, A, b, max_iter=100, tol=1e-6):
    """
    Solves a quadratic programming problem using gradient descent.
    Minimize: (1/2) x^T P x + q^T x
    Subject to: Gx <= h, Ax = b
    """
    n = len(q)
    x = [0 for _ in range(n)]  # Initial guess
    
    for iteration in range(max_iter):
        # Compute gradient
        grad = [sum(P[i][j] * x[j] for j in range(n)) + q[i] for i in range(n)]
        
        # Update x using gradient descent
        step_size = 0.01  # Learning rate
        x = [x[i] - step_size * grad[i] for i in range(n)]
        
        # Project onto constraints
        for i in range(len(G)):
            if dot_product(G[i], x) > h[i]:
                x = [x[j] - (dot_product(G[i], x) - h[i]) * G[i][j] for j in range(n)]
        
        # Check convergence
        if all(abs(grad[i]) < tol for i in range(n)):
            break
    
    return x

def svm_train_hard_margin(X, y):
    """
    Trains a linear SVM with hard margins using a custom QP solver.
    Hard margins are enforced by setting C to a very large value.
    """
    n_samples = len(X)
    n_features = len(X[0])
    
    # Define the QP problem matrices
    P = [[y[i] * y[j] * dot_product(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)]
    q = [-1 for _ in range(n_samples)]
    G = [[-1 if i == j else 0 for j in range(n_samples)] for i in range(n_samples)]  # Constraint: alphas >= 0
    h = [0 for _ in range(n_samples)]
    A = [y[i] for i in range(n_samples)]  # Constraint: sum(alpha_i * y_i) = 0
    b = [0]
    
    # Solve the QP problem
    alphas = solve_qp(P, q, G, h, [A], b)
    
    # Compute the weights and bias
    w = [0 for _ in range(n_features)]
    for i in range(n_samples):
        for j in range(n_features):
            w[j] += alphas[i] * y[i] * X[i][j]
    
    sv_indices = [i for i in range(n_samples) if alphas[i] > 1e-5]  # Support vectors
    b = sum(y[i] - dot_product(w, X[i]) for i in sv_indices) / len(sv_indices)
    
    return w, b

def svm_predict(w, b, X):
    """Predicts the labels for new data points."""
    predictions = []
    for x in X:
        prediction = 1 if dot_product(w, x) + b >= 0 else 0
        predictions.append(prediction)
    return predictions

# Main execution
if __name__ == "__main__":
    # Path to the dataset
    DATA_PATH = "/Users/Noura/Desktop/work/college/level 002/second sem/ML/Titanic project/data set/train.csv"

    # Load the data
    header, data = read_csv(DATA_PATH)
    
    # Preprocess the data
    X, y = preprocess_data(header, data)
    numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    n_numerical_features = len(numerical_features)
    # Scale the features
    X = scale_features(X,n_numerical_features)
    
    # Split into training and testing sets (80/20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train the SVM model with hard margins
    print("Training Hard-Margin SVM...")
    w, b = svm_train_hard_margin(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = svm_predict(w, b, X_test)
    accuracy = sum(1 for yp, yt in zip(y_pred, y_test) if yp == yt) / len(y_test)
    print(f"Hard-Margin SVM Model is trained")