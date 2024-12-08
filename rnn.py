import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# RNN + BiRNN to predict origin of name; 80 train 20 test 
# adapt tokenizer  for char; char pred in sequence s p9f 
# compare with ffnn

def to_number(labels):
    number_labels = []
    for label in labels:
        if label == 'Dutch':
            number_labels.append(0)
        elif label == 'Thai':
            number_labels.append(1)
        elif label == 'Russian':
            number_labels.append(2)
        elif label == 'Chinese':
            number_labels.append(3)
        elif label == 'Hawaiian':
            number_labels.append(4)
        elif label == 'Hungarian':
            number_labels.append(5)
    return number_labels



df = pd.read_csv("names.csv", sep=",")
X = df.name
y = df.nationality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train = X_train.tolist()
X_test = X_test.tolist()

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")


model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 300, input_length=MAX_LENGTH, trainable = True))
model.add(SimpleRNN(64, activation="relu", dropout=0.3, recurrent_dropout=0.3, return_sequences=TRUE, kernel_regularizer="l1_l2", bias_regularizer="l1_l2", activity_regularizer="l1_l2", recurrent_regularizer="l1_l2"))
model.add(Bidirectional(SimpleRNN(64, activation="relu", dropout=0.3, recurrent_dropout=0.3, return_sequences=TRUE, kernel_regularizer="l1_l2", bias_regularizer="l1_l2", activity_regularizer="l1_l2", recurrent_regularizer="l1_l2")))
model.add(Flatten())
model.add(Dense(64, activation="relu", kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-5)))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
model.summary()
model.fit(tokenized_X_train, y_train, validation_split=0.1, epochs=10, verbose=1, batch_size=20)

y_pred = model.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))