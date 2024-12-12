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
number_classes = len(y.unique())
y = np.array(to_number(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train = X_train.tolist()
X_test = X_test.tolist()

tokenizer = Tokenizer(char_level=True, lower=False)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1  # +1 to account for the padding token
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

model_rnn = Sequential()
model_rnn.add(Input(shape=(MAX_LENGTH,)))
model_rnn.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=MAX_LENGTH))
model_rnn.add(SimpleRNN(64, activation="relu"))
model_rnn.add(Dense(number_classes, activation="softmax"))
model_rnn.compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
model_rnn.fit(tokenized_X_train, y_train, validation_split=0.1, epochs=50, verbose=1, batch_size=20)

model_birnn = Sequential()
model_birnn.add(Input(shape=(MAX_LENGTH,)))
model_birnn.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=MAX_LENGTH))
model_birnn.add(Bidirectional(SimpleRNN(64, activation="relu")))
model_birnn.add(Dense(number_classes, activation="softmax"))
model_birnn.compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
model_birnn.fit(tokenized_X_train, y_train, validation_split=0.1, epochs=50, verbose=1, batch_size=20)

model_ffnn = Sequential()
model_ffnn.add(Input(shape=(MAX_LENGTH,)))
model_ffnn.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=MAX_LENGTH))
model_ffnn.add(Flatten())
model_ffnn.add(Dense(64, activation="relu"))
model_ffnn.add(Dense(number_classes, activation="softmax"))
model_ffnn.compile(loss="crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
model_ffnn.fit(tokenized_X_train, y_train, validation_split=0.1, epochs=50, verbose=1, batch_size=20)

y_pred = model_rnn.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))

y_pred = model_birnn.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))

y_pred = model_ffnn.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))
