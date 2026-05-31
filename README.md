# Iris Flower Classification using KNN

A simple machine learning project that classifies Iris flower species using the **K-Nearest Neighbors (KNN)** algorithm. It includes data preprocessing (scaling) and evaluates the model using accuracy, a confusion matrix, and a detailed classification report.

##  How It Works (The Pipeline)

1. **Dataset:** It loads the classic Iris dataset, which contains 150 samples of iris flowers across 3 distinct species (*setosa*, *versicolor*, and *virginica*).
2. **Data Splitting:** Splits the dataset into 80% for training the model and 20% for testing its performance.
3. **Feature Scaling:** Uses `StandardScaler` to normalize the data so that features with larger ranges don't unfairly bias the KNN distance math.
4. **Model Training:** Trains a `KNeighborsClassifier` using 5 neighbors ($n\_neighbors=5$).
5. **Evaluation:** Evaluates the predictions using:
   * **Confusion Matrix:** Shows exactly where the model predicted correctly and where it made mistakes.
   * **Accuracy Score:** The percentage of correct overall predictions.
   * **F1-Score:** Measures the balance between precision and recall, weighted across all three classes.

---

##  Requirements

Make sure you have `scikit-learn` and `pandas` installed:
```bash
pip install scikit-learn pandas

Bash
python iris_classification.py

Plaintext
Confusion Matrix : 
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
The Accuracy of  Confusion Matrix : 1.0

Weighted F1 Score = 1.0
The accuracy of F1 : 1.0

--- Detailed Classification Report ---
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
