from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('C:/Users/kavit/PycharmProjects/climate/models/fake_news_model.h5')

# Load the tokenizer
with open('C:/Users/kavit/PycharmProjects/climate/models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask app is running. Use POST /predict for prediction."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)
    return jsonify({'prediction': int(prediction > 0.3)})


if __name__ == '__main__':
    app.run(debug=True)
