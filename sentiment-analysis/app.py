from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('sentiment_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict([text])[0]
    result = "🟢 Positive" if prediction == 1 else "🔴 Negative"
    return render_template('index.html', result=result, text=text)

if __name__ == '__main__':
    app.run(debug=True)