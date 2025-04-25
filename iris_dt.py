import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import dagshub

dagshub.init(repo_owner='komalmca2003', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://github.com/komalmca2003/mlflow-dagshub-demo.git")

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
max_depth = 15
n_estimators = 10
mlflow.set_experiment('iris-dt')
with mlflow.start_run(run_name="pk_exp_with_confusion_matrix_log_artifact"):
        
    dt = DecisionTreeClassifier(max_depth = max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_param("max_depth", max_depth)

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")

    # Save the confusion matrix
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
        # Log the model
    mlflow.sklearn.log_model(dt, "decision_tree_model")

    mlflow.set_tag("author", "prakash")
    mlflow.set_tag("project", "iris-classification")
    mlflow.set_tag("algorithm", "decision-tree")

    

    print("Accuracy:", accuracy)

    