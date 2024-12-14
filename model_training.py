import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np

# Load the processed data
X_pad = np.load('C:/Users/kavit/PycharmProjects/climate/models/X.npy')
y = np.load('C:/Users/kavit/PycharmProjects/climate/models/Y.npy')

# Define the CNN model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: 0 or 1
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_pad, y, epochs=5, batch_size=64, validation_split=0.2)

# Save the model
model.save('C:/Users/kavit/PycharmProjects/climate/models/fake_news_model.h5')

print("Model training complete and saved!")
