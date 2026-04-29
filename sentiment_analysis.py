import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

def load_dataset():
    """1. Generate a synthetic labeled sentiment dataset."""
    positive_words = ["love", "great", "fantastic", "amazing", "beautiful", "superb", "brilliant", "excellent", "delightful", "uplifting", "good", "best", "wonderful", "masterpiece", "joy", "captivating", "stunning"]
    negative_words = ["hate", "terrible", "boring", "awful", "poor", "dull", "disaster", "horrible", "worst", "pain", "bad", "ugly", "depressing", "confusing", "disappointing", "uninteresting"]
    neutral_words = ["movie", "film", "acting", "plot", "story", "director", "visuals", "experience", "performance", "watch", "character", "scene", "the", "a", "an", "and", "but", "is", "was", "it"]

    np.random.seed(42)
    data = []
    
    # Generate 1000 samples
    for _ in range(1000):
        sentiment = np.random.choice([0, 1]) # 0 for negative, 1 for positive
        
        # Build sentences
        if sentiment == 1:
            words = np.random.choice(positive_words, size=np.random.randint(2, 6)).tolist()
        else:
            words = np.random.choice(negative_words, size=np.random.randint(2, 6)).tolist()
        
        words += np.random.choice(neutral_words, size=np.random.randint(4, 10)).tolist()
        np.random.shuffle(words)
        
        sentence = " ".join(words) + "."
        label = "Positive" if sentiment == 1 else "Negative"
        data.append({"text": sentence, "label": label})
        
    return pd.DataFrame(data)

def preprocess_text(text):
    """Clean text: lowercase, remove punctuation, remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

def perform_eda(df):
    """3. EDA: Print dataset shape, class distribution, sample rows, and top 10 words."""
    print("--- Exploratory Data Analysis ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    
    print("\nSample Rows:")
    print(df.head())
    
    # Top 10 words for positive and negative
    pos_reviews = df[df['label'] == 'Positive']['cleaned_text']
    neg_reviews = df[df['label'] == 'Negative']['cleaned_text']
    
    pos_words = " ".join(pos_reviews).split()
    neg_words = " ".join(neg_reviews).split()
    
    pos_word_counts = pd.Series(pos_words).value_counts().head(10)
    neg_word_counts = pd.Series(neg_words).value_counts().head(10)
    
    print("\nTop 10 Most Frequent Words - Positive Reviews:")
    print(pos_word_counts)
    
    print("\nTop 10 Most Frequent Words - Negative Reviews:")
    print(neg_word_counts)
    print("-" * 35 + "\n")

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """4 & 5. Train models, evaluate, and return the best one."""
    print("--- Model Training & Evaluation ---")
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    # Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    
    # Evaluate
    def get_metrics(y_true, y_pred, pos_label='Positive'):
        acc = accuracy_score(y_true, y_pred)
        # Using pos_label for binary classification metrics
        prec = precision_score(y_true, y_pred, pos_label=pos_label)
        rec = recall_score(y_true, y_pred, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label)
        return acc, prec, rec, f1

    lr_acc, lr_prec, lr_rec, lr_f1 = get_metrics(y_test, lr_preds)
    nb_acc, nb_prec, nb_rec, nb_f1 = get_metrics(y_test, nb_preds)
    
    print("Logistic Regression Metrics:")
    print(f"Accuracy: {lr_acc:.4f}, Precision: {lr_prec:.4f}, Recall: {lr_rec:.4f}, F1-score: {lr_f1:.4f}")
    
    print("\nMultinomial Naive Bayes Metrics:")
    print(f"Accuracy: {nb_acc:.4f}, Precision: {nb_prec:.4f}, Recall: {nb_rec:.4f}, F1-score: {nb_f1:.4f}")
    
    # Determine best model based on Accuracy (or F1-score)
    if lr_acc >= nb_acc:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_preds = lr_preds
    else:
        best_model = nb_model
        best_name = "Multinomial Naive Bayes"
        best_preds = nb_preds
        
    print(f"\nBest Model: {best_name}")
    print(f"Reason: It achieved a higher accuracy ({max(lr_acc, nb_acc):.4f}) on the test set.")
    
    print("\nConfusion Matrix for Best Model:")
    print(confusion_matrix(y_test, best_preds, labels=['Negative', 'Positive']))
    print("-" * 35 + "\n")
    
    return best_model

def save_model(model, vectorizer, filename="sentiment_model.pkl"):
    """6. Save the model and vectorizer."""
    joblib.dump({'model': model, 'vectorizer': vectorizer}, filename)
    print(f"Model and vectorizer saved to {filename}")

def predict_sentiment(text, filename="sentiment_model.pkl"):
    """7. Load model, preprocess, and predict sentiment for new text."""
    data = joblib.load(filename)
    model = data['model']
    vectorizer = data['vectorizer']
    
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction

def main():
    # 1. Dataset
    df = load_dataset()
    
    # 2. Preprocessing
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # 3. EDA
    perform_eda(df)
    
    # Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4 & 5. Model Training and Evaluation
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 6. Save the Model
    save_model(best_model, vectorizer)
    
    # 7. Test It
    print("\n--- Testing predict_sentiment() ---")
    sample_sentences = [
        "I absolutely loved this movie, it was a fantastic masterpiece!",
        "This was a terrible and boring film, worst acting ever.",
        "The visuals were stunning, but the plot was completely confusing.",
        "The acting was superb and the story was very gripping.",
        "It was a total disaster and a complete waste of time."
    ]
    
    for sentence in sample_sentences:
        pred = predict_sentiment(sentence)
        print(f"Sentence: '{sentence}' -> Predicted Sentiment: {pred}")

if __name__ == "__main__":
    main()
