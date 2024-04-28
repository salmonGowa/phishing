import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (e.g., phishing email dataset)
data = pd.read_csv('phishing_emails.csv')

# Split dataset into features (X) and target (y)
X = data['email_content']
y = data['label']  # 'legitimate' or 'phishing'

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Real-time detection (example)
def predict_phishing_email(email_content):
    email_vector = vectorizer.transform([email_content])
    prediction = model.predict(email_vector)
    if prediction[0] == 'phishing':
        return "This email is likely a phishing attempt."
    else:
        return "This email appears to be legitimate."

# Example usage
email_content = "Click this link to verify your account: www.fakebank.com/verify"
result = predict_phishing_email(email_content)
print(result)
