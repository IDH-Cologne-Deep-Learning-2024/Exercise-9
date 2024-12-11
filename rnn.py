import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, SimpleRNN, Bidirectional, LSTM, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def to_number(labels):
    label_map = {
        'Dutch': 0,
        'Thai': 1,
        'Russian': 2,
        'Chinese': 3,
        'Hawaiian': 4,
        'Hungarian': 5
    }
    return [label_map[label] for label in labels]

df = pd.read_csv("names.csv", sep=",")
df['label'] = to_number(df['label'])

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['name'])
X = tokenizer.texts_to_sequences(df['name'])
X = pad_sequences(X, padding='post')

y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

embedding_dim = 50
num_classes = len(set(y))
input_length = X.shape[1]

rnn_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=input_length),
    SimpleRNN(128, activation='tanh', return_sequences=False),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

birnn_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=input_length),
    Bidirectional(SimpleRNN(128, activation='tanh', return_sequences=False)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

birnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
birnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

ffnn_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=input_length),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

ffnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ffnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

def evaluate_model(model, X_test, y_test, model_name):
    predictions = np.argmax(model.predict(X_test), axis=-1)
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, predictions, target_names=list(tokenizer.word_index.keys())))

evaluate_model(rnn_model, X_test, y_test, "RNN")

evaluate_model(birnn_model, X_test, y_test, "BiRNN")

evaluate_model(ffnn_model, X_test, y_test, "FFNN")
