# 🚢 Titanic 1912: The Sink-O-Matic™ & Machine Learning Suite

> *"Billed as the absolute pinnacle of unsinkable engineering... until Machine Learning audited the passenger manifest."*

An end-to-end Machine Learning project featuring algorithms implemented **from scratch** in pure Python (Logistic Regression, Linear Regression, and custom SVM Kernels) alongside **Scikit-Learn** benchmark models, bundled with a sarcastic, dark-ocean themed interactive **Streamlit** frontend.

---

## 🌟 Highlights & Features

- 🧠 **ML From Scratch**:
  - Custom **Logistic Regression** with gradient descent & L2 (Ridge) regularization.
  - Custom **Linear Regression** with Batch Gradient Descent, SGD, and Ridge modes.
  - Custom **Support Vector Machines (SVM)**: Hard Margins, Polynomial Kernel, Quadratic Kernel, and Radial Basis Function (RBF) Kernel.
- ⚖️ **Model Benchmark & Face-Off**:
  - Compares scratch-built models against optimized Scikit-Learn baselines.
- 🎨 **Sarcastic Streamlit Frontend (`app.py`)**:
  - **💀 Sarcastic Survival Oracle**: Calculates survival odds with dynamic historical roasts based on class, sex, age, and fare.
  - **📜 1912 Manifest & Inquest Certificate**: Generates official White Star Line casualty or survivor certificate cards.
  - **💸 Extortion Fare Appraiser**: Predicts ticket pricing with 1912 £ to modern USD inflation conversions.
  - **🚪 The Rose's Door Physics Lab**: Interactive buoyancy and selfishness quotient simulator to answer if Jack really had to freeze.
  - **📊 Feature Weights & Demographics**: Interprets learned weights from gradient descent.

---

## 🗂️ Project Structure

```
├── app.py                     # Streamlit web application & interactive UI
├── userPredict2.py            # Unified predictor class bridging models & preprocessing
├── Finalclassification.py     # Custom Logistic Regression from scratch & Sklearn benchmark
├── Finalregression.py         # Custom Linear Regression (Batch/SGD/Ridge) from scratch
├── HardMargins.py             # Custom Hard-Margin SVM implementation
├── PolynomialKernel.py        # Custom Polynomial Kernel SVM implementation
├── Quadratic.py               # Custom Quadratic Kernel SVM implementation
├── Rbf.py                     # Custom RBF Kernel SVM implementation
├── data_loading.py            # Robust dataset loader and parser
├── data set/                  # Dataset folder
│   ├── train.csv              # Titanic training dataset
│   ├── test.csv               # Titanic test dataset
│   └── gender_submission.csv  # Baseline benchmark dataset
├── .gitignore                 # Git ignore rules for cache and virtual environments
└── README.md                  # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nrshousha/ML-project-Titanic-Survival-.git
cd ML-project-Titanic-Survival-
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install streamlit scikit-learn numpy pandas
```

---

## 🚀 Running the Project

### Option A: Launch the Streamlit Interactive Frontend
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the UI.

### Option B: Run CLI Prediction
```bash
python userPredict2.py
```

### Option C: Train Individual Models
```bash
# Classification models
python Finalclassification.py

# Regression models
python Finalregression.py
```

---

## 📊 Model Details

### 1. Classification (Survival Prediction)
- **Features Used**: `[Pclass, Sex, Age, Fare]`
- **Scaling**: `StandardScaler`
- **Custom Model**: Sigmoid activation with manual Ridge loss:
  $$\hat{y} = \sigma(W^T X + b)$$
  $$w_j \leftarrow w_j - \alpha \left( (y_{pred} - y) \cdot x_j + \lambda w_j \right)$$
- **Baseline**: Scikit-Learn Logistic Regression (`saga` solver, $C=0.5$).

### 2. Regression (Fare Prediction)
- **Features Used**: `[Pclass, Sex, Age, SibSp, Parch]`
- **Scaling**: `StandardScaler`
- **Custom Model**: Linear Regression with Batch / Stochastic Gradient Descent & Ridge penalty:
  $$\hat{y} = W^T X + b$$
- **Baseline**: Scikit-Learn Ridge Regression ($\alpha=0.5$).

### 3. Kernel SVMs
- Hard-Margin Linear SVM
- Polynomial Kernel: $K(x, x') = (x \cdot x' + c)^d$
- Quadratic Kernel: $K(x, x') = (x \cdot x' + 1)^2$
- Radial Basis Function (RBF) Kernel: $K(x, x') = \exp(-\gamma ||x - x'||^2)$

---

## 📜 License & Academic Context
Created for College Machine Learning coursework. Open source under the MIT License.
