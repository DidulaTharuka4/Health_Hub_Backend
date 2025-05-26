import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# === Load Dataset ===
df = pd.read_csv('data/face_emotions.csv')  # columns: emotion_sequence, overall_emotion
df['emotion_sequence'] = df['emotion_sequence'].astype(str)

# === Vectorize Sequences ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['emotion_sequence'])
y = df['overall_emotion']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("Face Emotion Aggregator Report:")
print(classification_report(y_test, y_pred))

# === Save Model and Vectorizer ===
pickle.dump(model, open("models/saved_aggregator_model_face.pkl", "wb"))
pickle.dump(vectorizer, open("models/saved_aggregator_vectorizer_face.pkl", "wb"))