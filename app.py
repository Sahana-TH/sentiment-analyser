from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

pipeline = joblib.load('sentiment_model.pkl')

# Store last 5 predictions
history = []

@app.route('/')
def home():
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    prediction = pipeline.predict([text])[0]
    confidence = pipeline.predict_proba([text])[0]
    confidence_score = round(max(confidence) * 100, 1)
    
    if prediction == 1:
        result = "🟢 Positive"
        color = "green"
    else:
        result = "🔴 Negative"
        color = "red"

    # Add to history (keep last 5 only)
    history.insert(0, {
        'text': text,
        'result': result,
        'confidence': confidence_score,
        'color': color
    })
    if len(history) > 5:
        history.pop()

    return render_template('index.html',
                           result=result,
                           confidence=confidence_score,
                           color=color,
                           text=text,
                           history=history)

if __name__ == '__main__':
    app.run(debug=True)