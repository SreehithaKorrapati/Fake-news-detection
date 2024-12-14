import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model('C:/Users/kavit/PycharmProjects/climate/models/fake_news_model.h5')

# Load the tokenizer
with open('C:/Users/kavit/PycharmProjects/climate/models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Load test data (optional, if you have test data)
X_test = np.load('C:/Users/kavit/PycharmProjects/climate/models/X.npy')
y_test = np.load('C:/Users/kavit/PycharmProjects/climate/models/Y.npy')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Example prediction
new_text = ["This is a test news article"]
new_seq = tokenizer.texts_to_sequences(new_text)
new_pad = pad_sequences(new_seq, maxlen=100)

prediction = model.predict(new_pad)
print("Prediction (0 = Fake, 1 = Real):", int(prediction > 0.5))
