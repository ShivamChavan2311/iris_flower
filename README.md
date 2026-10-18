<h1 align="center">🌸 Iris Flower Classification</h1>

<p align="center">
  <i>Predicting Iris species from flower measurements with KNN, Logistic Regression and Naive Bayes.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

---

## 📋 Problem Statement

A botanical research centre identifies Iris species by manually measuring sepal and petal dimensions. This process is:

- **Time-consuming:** manual measurement takes hours
- **Error-prone:** human mistakes creep into identification
- **Not scalable:** it cannot handle large datasets

**Solution:** build automated ML classifiers that predict the species directly from the four physical measurements.

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Iris Dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) |
| Samples | 150 (50 per class) |
| Classes | 3: *Iris-setosa*, *Iris-versicolor*, *Iris-virginica* |
| Features | 4 numerical measurements |
| Missing values | None |

| Feature | Description |
|---|---|
| `SepalLengthCm` | Length of sepal (cm) |
| `SepalWidthCm` | Width of sepal (cm) |
| `PetalLengthCm` | Length of petal (cm) |
| `PetalWidthCm` | Width of petal (cm) |
| `Species` (target) | Iris-setosa, Iris-versicolor, Iris-virginica |

**Characteristics:** clean, perfectly balanced, no missing values, well-separated classes. ⚠️ It is a small dataset, so results should be read with that in mind.

## 🤖 Models

| Model | How it works | Why use it |
|---|---|---|
| **K-Nearest Neighbors** | Finds the *k* closest training samples and votes on the class | Simple, no training phase, fast to set up |
| **Logistic Regression** | Learns linear decision boundaries and class probabilities | Interpretable and efficient |
| **Gaussian Naive Bayes** | Applies Bayes' theorem, assuming features are independent | Very fast to train and predict |

## 📈 Results

Evaluated on a held-out **50% test set (75 samples)** with `random_state=12`.

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| KNN | **100%** | 100% | 100% |
| Logistic Regression | 98.67% | 98.67% | 98.67% |
| Naive Bayes | **100%** | 100% | 100% |

🏆 **Best models:** KNN and Naive Bayes (tied at 100% accuracy). Logistic Regression misclassified a single test sample (74/75).

> **Note:** Iris is a very easy dataset, so scores this high are expected. Results on a 75-sample test set can shift with a different split, so treat the ranking between models as indicative rather than definitive.

<!--
Add your charts here once they are in the repo:
<p align="center">
  <img src="results/model_comparison.png" width="48%" />
  <img src="results/confusion_matrices.png" width="48%" />
</p>
-->

## 🛠️ Installation

**Prerequisites:** Python 3.7+, `pandas`, `numpy`, `scikit-learn`

```bash
# Clone the repository
git clone https://github.com/ShivamChavan2311/iris-flower-classification.git
cd iris-flower-classification

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**

```
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
```

## 🚀 Usage

### 1. Load and prepare the data

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Iris.csv")

# Encode species labels to numbers (0, 1, 2)
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Separate features and target (drops the Id column if your CSV has one)
X = df.drop(columns=["Id", "Species"], errors="ignore")
Y = df["Species"]

# 50-50 train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=12
)

# Scale features (fit on train only to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Train the models

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, Y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, Y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, Y_train)
y_pred_nb = nb.predict(X_test_scaled)
```

### 3. Evaluate

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(f"KNN Accuracy: {accuracy_score(Y_test, y_pred_knn):.4f}")
print(f"LR  Accuracy: {accuracy_score(Y_test, y_pred_lr):.4f}")
print(f"NB  Accuracy: {accuracy_score(Y_test, y_pred_nb):.4f}")

print(confusion_matrix(Y_test, y_pred_knn))
print(classification_report(Y_test, y_pred_knn, target_names=le.classes_))
```

## 🔑 Pipeline at a Glance

1. **Load data:** read the CSV into a pandas DataFrame
2. **Encode target:** convert species names to 0, 1, 2
3. **Separate features and target:** `X` = measurements, `Y` = species
4. **Train-test split:** 50% train, 50% test
5. **Scale features:** `StandardScaler`, fitted on the training set only
6. **Train models:** fit all three classifiers
7. **Evaluate:** compare accuracy, precision, recall and F1, and inspect confusion matrices

## 📁 Project Structure

```
iris-flower-classification/
│
├── Iris.csv                    # Dataset
├── iris_classification.ipynb   # Notebook with full code
├── iris_classification.py      # Python script version
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
└── results/
    ├── model_comparison.png    # Performance comparison chart
    └── confusion_matrices.png  # Confusion matrices
```

## 📚 What I Learned

- Data preprocessing: label encoding, feature scaling and train-test splitting
- Training and comparing several classification algorithms
- Reading evaluation metrics: accuracy, precision, recall, F1, confusion matrix
- Why distance-based models like KNN need scaled features
- Why a clean, balanced dataset gives near-perfect scores, and why real data is harder

## ⚠️ Important Notes

- This dataset is clean and balanced, so near-perfect accuracy is expected. Real-world data is messier and needs more preprocessing.
- Always scale features before using distance-based models such as KNN.
- Fit the scaler on the training set only, then apply it to the test set, to avoid data leakage.
- Compare several metrics, not just accuracy.

## 🔭 Future Improvements

- Use stratified splits and k-fold cross-validation for more reliable comparisons
- Tune hyperparameters (for example, *k* in KNN) with `GridSearchCV`
- Try more models such as SVM and Random Forest
- Visualise decision boundaries with PCA

## 📖 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Iris Dataset (UCI)](https://archive.ics.uci.edu/dataset/53/iris)
- [Supervised Learning Algorithms](https://scikit-learn.org/stable/supervised_learning.html)

## 👤 Author

**Shivam Chavan**: Supervised Machine Learning, Assignment 3

[LinkedIn](https://www.linkedin.com/in/shivamchavansppu/) · [Portfolio](https://shivamchavan2311.github.io/portfolio/) · [GitHub](https://github.com/ShivamChavan2311) · [chavanshivam160@gmail.com](mailto:chavanshivam160@gmail.com)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Feel free to fork the repo, make changes and open a pull request.

---

<p align="center"><b>Status: Complete ✅</b></p>

