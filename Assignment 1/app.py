from flask import Flask, request, jsonify
import pickle

# Load the saved model and vectorizer
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input with key 'text'
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess and vectorize the input text
    text = data['text']
    text_vectorized = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_vectorized)
    sentiment = 'positive' if prediction[0] == 1 else 'negative'

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
