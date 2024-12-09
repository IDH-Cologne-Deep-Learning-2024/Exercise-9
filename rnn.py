import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Bidirectional, Embedding, Flatten, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Set the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load and preprocess the data
df = pd.read_csv("names.csv")

# Tokenize names at the character level
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['name'])
X = tokenizer.texts_to_sequences(df['name'])
X = pad_sequences(X, padding='post')

# Convert nationalities to numeric labels
nationality_mapping = {n: i for i, n in enumerate(df['nationality'].unique())}
y = df['nationality'].map(nationality_mapping).values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper function to build models
def build_rnn_model(input_dim, output_dim, rnn_units, num_classes, bidirectional=False):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=X.shape[1]))
    if bidirectional:
        model.add(Bidirectional(SimpleRNN(rnn_units, return_sequences=False)))
    else:
        model.add(SimpleRNN(rnn_units, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Helper function to build FFNN
def build_ffnn_model(input_dim, output_dim, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=X.shape[1]))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Define hyperparameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50
rnn_units = 64
num_classes = len(nationality_mapping)

# Train and evaluate RNN
rnn_model = build_rnn_model(vocab_size, embedding_dim, rnn_units, num_classes)
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32)
rnn_predictions = rnn_model.predict(X_test)
print("RNN Classification Report")
print(classification_report(y_test, np.argmax(rnn_predictions, axis=1)))

# Train and evaluate BiRNN
birnn_model = build_rnn_model(vocab_size, embedding_dim, rnn_units, num_classes, bidirectional=True)
birnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
birnn_model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32)
birnn_predictions = birnn_model.predict(X_test)
print("BiRNN Classification Report")
print(classification_report(y_test, np.argmax(birnn_predictions, axis=1)))

# Train and evaluate FFNN
ffnn_model = build_ffnn_model(vocab_size, embedding_dim, num_classes)
ffnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ffnn_model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32)
ffnn_predictions = ffnn_model.predict(X_test)
print("FFNN Classification Report")
print(classification_report(y_test, np.argmax(ffnn_predictions, axis=1)))
