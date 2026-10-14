from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from svm_model import optimize_svm, predict_svm


# Load sample data
data = load_iris()

X = data.data
y = data.target


# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Optimize SVM
model = optimize_svm(X_train, y_train)


# Make predictions
y_pred = predict_svm(model, X_test)


print("Predictions:", y_pred)