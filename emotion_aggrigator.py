import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# === Load dataset ===
# Each row: emotion_sequence: "happy,sad,neutral,happy", overall_emotion: "happy"
df = pd.read_csv('data/face_emotion_logs.csv')  # or content_emotion_logs.csv

# === Convert list of emotions into string and use CountVectorizer to extract features ===
X_raw = df['emotion_sequence']
y = df['overall_emotion']

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
X = vectorizer.fit_transform(X_raw)

# === Train Model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === Save Model and Vectorizer ===
with open('models/saved_aggregator_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/saved_aggregator_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
