import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Flatten,
    Input,
    Dense,
    Dropout,
    SimpleRNN,
    Bidirectional,
    LSTM,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def to_number(labels):
    number_labels = []
    for label in labels:
        if label == "Dutch":
            number_labels.append(0)
        elif label == "Thai":
            number_labels.append(1)
        elif label == "Russian":
            number_labels.append(2)
        elif label == "Chinese":
            number_labels.append(3)
        elif label == "Hawaiian":
            number_labels.append(4)
        elif label == "Hungarian":
            number_labels.append(5)
    return number_labels


df = pd.read_csv("names.csv", sep=",")
df["nationality"] = to_number(df["nationality"])

X = df["name"]
y = df["nationality"]

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X)
vocab_size = len(tokenizer.word_index) + 1
max_name_length = max([len(name) for name in X])
num_classes = len(set(y))
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, padding="post")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# RNN
RNN = Sequential(
    [
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50),
        SimpleRNN(128, activation="tanh", return_sequences=False),
        Dropout(0.5),
        Dense(
            64,
            activation="tanh",
            kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=L2(1e-4),
        ),
        Dense(num_classes, activation="softmax"),
    ]
)

RNN.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
RNN.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

y_pred = RNN.predict(X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))

# BiRNN
BIRNN = Sequential(
    [
        Embedding(
            input_dim=len(tokenizer.word_index) + 1,
            output_dim=50,
            input_length=X.shape[1],
        ),
        Bidirectional(LSTM(units=128, return_sequences=False)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ]
)

BIRNN.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
BIRNN.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

y_pred = BIRNN.predict(X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))

# FFNN
#My FFNN crashes on the fit function because of a dimension difference i cant solve.
FFNN = Sequential(
    [
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100),
        Dense(128, activation="tanh"),
        Dropout(0.5),
        Dense(
            64,
            activation="tanh",
            kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=L2(1e-4),
        ),
        Dense(num_classes, activation="softmax"),
    ]
)

FFNN.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
FFNN.fit((X_train, y_train), epochs=25, batch_size=32, validation_split=0.2)

print("FFNN results:")
y_pred = FFNN.predict(X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))

print("RNN results again:")
y_pred = RNN.predict(X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))
