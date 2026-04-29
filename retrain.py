import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

print("Loading dataset...")
# Use positive/negative categories only
categories_pos = ['rec.sport.hockey', 'rec.autos', 'sci.space', 'talk.politics.misc']
categories_neg = ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian', 'sci.med']

train_pos = fetch_20newsgroups(subset='train', categories=categories_pos)
train_neg = fetch_20newsgroups(subset='train', categories=categories_neg)
test_pos = fetch_20newsgroups(subset='test', categories=categories_pos)
test_neg = fetch_20newsgroups(subset='test', categories=categories_neg)

# Combine and label data
X_train = train_pos.data + train_neg.data
y_train = [0]*len(train_pos.data) + [1]*len(train_neg.data)
X_test = test_pos.data + test_neg.data
y_test = [0]*len(test_pos.data) + [1]*len(test_neg.data)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("Training model...")
# Build a proper pipeline with vectorizer + model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver='lbfgs'
    ))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Negative', 'Positive']))

# Test with sample sentences
print("\n--- Testing Sample Sentences ---")
samples = [
    "I love this product, it is amazing!",
    "This is absolutely terrible and awful",
    "What a wonderful experience!",
    "I hate this, worst thing ever",
    "Pretty good, I enjoyed it"
]
for s in samples:
    pred = pipeline.predict([s])[0]
    label = "🟢 Positive" if pred == 1 else "🔴 Negative"
    print(f"{label} → {s}")

# Save the new improved model as a pipeline
print("\nSaving improved model...")
joblib.dump(pipeline, 'sentiment_model.pkl')
print("✅ Model saved as sentiment_model.pkl")