import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression, Ridge
from Finalregression import LinearRegression as CustomLinearRegression
from Finalregression import preprocess_regression
from Finalclassification import LogisticRegression as CustomLogisticRegression
from Finalclassification import preprocess_classification
from data_loading import read_csv

class TitanicPassengerPredictor:

    def __init__(self):
        # Load preprocessed data for classification and regression
        header, data = read_csv("train.csv")
        
        # Preprocess data for classification
        self.X_class, self.y_class = preprocess_classification(data)
        self.scaler_class = StandardScaler()
        self.X_class_scaled = self.scaler_class.fit_transform(self.X_class)
        
        # Train classification models (Custom and Scikit-Learn)
        self.custom_classi_model = CustomLogisticRegression(learning_rate=0.01, epochs=1000, mode='ridge', lmbd=0.1)
        self.custom_classi_model.train(self.X_class_scaled.tolist(), self.y_class)  # Convert to list for compatibility
        
        self.sklearn_classi_model = SklearnLogisticRegression(
            penalty='l2',
            solver='saga',
            C=0.5,
            max_iter=1000
        ).fit(self.X_class_scaled, self.y_class)

        # Preprocess data for regression
        self.X_reg, self.y_reg = preprocess_regression(data)
        self.scaler_reg = StandardScaler()
        self.X_reg_scaled = self.scaler_reg.fit_transform(self.X_reg)
        
        # Train regression models (Custom and Scikit-Learn)
        self.custom_reg_model = CustomLinearRegression(learning_rate=0.0001, epochs=1000, mode='ridge', lmbd=0.1)
        self.custom_reg_model.train(self.X_reg_scaled.tolist(), self.y_reg)  # Convert to list for compatibility
        
        self.sklearn_reg_model = Ridge(alpha=0.5).fit(self.X_reg_scaled, self.y_reg)

    def predict_new(self):
        print('\n=== Titanic Prediction ===\n')

        while True:
            # Choosing classification or regression
            choice = input('\nPredict: \n1. Classification (Survived or Died)\n2. Regression (Fare Prediction)\nChoice (1/2): ')

            if choice == '1':
                # User inputs for classification
                pclass = int(input("Passenger class (choose 1, 2, or 3): "))
                sex = 1 if input('Sex (male or female): ').lower() == 'male' else 0
                age = float(input('Age: '))
                fare = float(input('Paid Fare: '))

                # Scale inputs
                passenger_data = self.scaler_class.transform([[pclass, sex, age, fare]])

                # Predict survival (classification) using both models
                print("\n--- Classification Predictions ---")
                
                # Custom Logistic Regression
                custom_prediction = self.custom_classi_model.predict(passenger_data.tolist())[0]
                custom_prob = self.sigmoid(sum([self.custom_classi_model.weights[j] * passenger_data[0][j] for j in range(len(passenger_data[0]))]) + self.custom_classi_model.bias)
                print(f"Custom Model: {'Survived' if custom_prediction == 1 else 'Died'} ({custom_prob * 100:.2f}% probability)")
                
                # Scikit-Learn Logistic Regression
                sklearn_prediction = self.sklearn_classi_model.predict(passenger_data)[0]
                sklearn_prob = self.sklearn_classi_model.predict_proba(passenger_data)[0][1]
                print(f"Scikit-Learn Model: {'Survived' if sklearn_prediction == 1 else 'Died'} ({sklearn_prob * 100:.2f}% probability)")
                
                break

            elif choice == '2':
                # User inputs for regression
                pclass = int(input("Passenger class (choose 1, 2, or 3): "))
                sex = 1 if input('Sex (male or female): ').lower() == 'male' else 0
                age = float(input('Age: '))
                sibsp = int(input('Number of siblings/spouses aboard: '))
                parch = int(input('Number of parents/children aboard: '))

                # Scale inputs
                passenger_data = self.scaler_reg.transform([[pclass, sex, age, sibsp, parch]])

                # Predict fare (regression) using both models
                print("\n--- Regression Predictions ---")
                
                # Custom Linear Regression
                custom_fare = self.custom_reg_model.predict(passenger_data.tolist())[0]  # Predict fare
                print(f"Custom Model: ${custom_fare:.2f}")
                
                # Scikit-Learn Ridge Regression
                sklearn_fare = self.sklearn_reg_model.predict(passenger_data)[0]  # Predict fare
                print(f"Scikit-Learn Model: ${sklearn_fare:.2f}")
                
                break

            else:
                print('Invalid choice. Try again.')

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + math.exp(-z))


if __name__ == '__main__':
    predictor = TitanicPassengerPredictor()
    while True:
        predictor.predict_new()
        if input('\nPredict another passenger? (y/n): ').lower() != 'y':
            break