# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Data Loading
train_data = pd.read_csv("C:/Users/rovho/Downloads/train.csv")
test_data = pd.read_csv("C:/Users/rovho/Downloads/test.csv")

# Exploratory Data Analysis
print("Training Data:")
print(train_data.head())

print("\nTest Data:")
print(test_data.head())

# Survival Rate Analysis
rate_women = sum(train_data.loc[train_data.Sex == 'female']["Survived"]) / len(train_data.loc[train_data.Sex == 'female'])
rate_men = sum(train_data.loc[train_data.Sex == 'male']["Survived"]) / len(train_data.loc[train_data.Sex == 'male'])

print("\n% of women who survived:", rate_women)
print("\n% of men who survived:", rate_men)

#

# Model Training
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
y = train_data["Survived"]
X_test = pd.get_dummies(test_data[features])


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# Results Saving
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv("C:/Users/rovho/OneDrive/Downloads/Documents/results.csv", index=False)
print("\nYour Results file was successfully saved!")

# Model Evaluation
ground_truth = pd.read_csv("C:/Users/rovho/OneDrive/Downloads/Documents/results.csv")
y_true = ground_truth['Survived']


# Evaluate the performance of the model
accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions)
recall = recall_score(y_true, predictions)
f1 = f1_score(y_true, predictions)
conf_matrix = confusion_matrix(y_true, predictions)

# Display evaluation metrics
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Display the confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)
