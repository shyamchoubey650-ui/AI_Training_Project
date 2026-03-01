import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("C:\\Users\\shyam\\OneDrive\\Desktop\\AI_Training_Project (4th sem)\\AI_Training_Project\\data\\combined_data.csv")  # <-- use correct filename

print("Columns:", data.columns)
print("Dataset size:", data.shape)

# Clean label column
data['label'] = data['label'].astype(str).str.lower().str.strip()

# Convert labels to numeric
data['label'] = data['label'].replace({
    'spam': 1,
    'ham': 0,
    'not spam': 0
})

# Remove invalid rows
data = data.dropna()
data = data[data['text'].str.strip() != ""]

print("\nClass Distribution:")
print(data['label'].value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel saved successfully!")