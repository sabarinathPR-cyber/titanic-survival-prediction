import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("titanic_dataset.csv")

# Select features
data = data[['Pclass','Sex','Age','Fare','Survived']]

# Convert gender to numbers
data['Sex'] = data['Sex'].map({'male':0,'female':1})

# Handle missing values
data = data.dropna()

X = data[['Pclass','Sex','Age','Fare']]
y = data['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
