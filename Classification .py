import sklearn 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report

iris = load_iris()
x = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.20, 
    random_state=42, 
    shuffle=True
)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = KNeighborsClassifier(n_neighbors=5)


model.fit(x_train, y_train)


predictions = model.predict(x_test)


cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix : ")
print(cm)
print()

f1 = f1_score(y_test, predictions, average="weighted")
print(f"Weighted F1 Score = {f1 : .4f}")
print()

print("--- Detailed Classification Report ---")
print(classification_report(y_test, predictions, target_names=iris.target_names))
