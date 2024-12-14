import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np

# Load datasets
true_news = pd.read_csv('C:/Users/kavit/PycharmProjects/climate/data/fake.csv')
fake_news = pd.read_csv('C:/Users/kavit/PycharmProjects/climate/data/true.csv')

# Preprocessing: Combine true and fake news into one dataset
true_news['label'] = 1  # Real news -> 1
fake_news['label'] = 0  # Fake news -> 0

# Concatenate datasets
data = pd.concat([true_news[['text', 'label']], fake_news[['text', 'label']]])

# Clean the text (remove special characters, unnecessary spaces)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

data['text'] = data['text'].apply(clean_text)

# Split into features (X) and labels (y)
X = data['text']
y = data['label']

# Tokenize and pad the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)

# Save the tokenizer for later use
with open('C:/Users/kavit/PycharmProjects/climate/models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)

# Save the processed data to be used for training
np.save('C:/Users/kavit/PycharmProjects/climate/models/X.npy', X_pad)
np.save('C:/Users/kavit/PycharmProjects/climate/models/Y.npy', y)

print("Preprocessing complete and data saved!")
