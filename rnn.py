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

def RNN():
    df = pd.read_csv("names.csv", sep=",")
    df = df.head(1000)
    X = df.name
    y = to_number(df.nationality)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)

    X_train = X_train.to_list()
    X_test = X_test.to_list()

    tokenizer = Tokenizer(num_words=None, lower=True, split=' ', char_level=True, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    tokenized_X_train = tokenizer.texts_to_sequences(X_train)
    tokenized_X_test = tokenizer.texts_to_sequences(X_test)
    MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
    tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding='post')
    tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding='post')
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=MAX_LENGTH))
    model.add(SimpleRNN(64, input_shape = (MAX_LENGTH, 1), activation="relu", dropout=0.2, recurrent_dropout=0.3, return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4), recurrent_regularizer=L2(1e-5)))
    model.add(Bidirectional(SimpleRNN(64)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(tokenized_X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)
    y_pred = model.predict(tokenized_X_test)
    y_pred = y_pred.argmax(axis=1)
    return classification_report(y_test, y_pred)

def FFNN():
    df = pd.read_csv("names.csv", sep=",")
    df = df.head(1000)
    X = df.name
    y = to_number(df.nationality)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)

    X_train = X_train.to_list()
    X_test = X_test.to_list()

    tokenizer = Tokenizer(num_words=None, lower=True, split=' ', char_level=True, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    tokenized_X_train = tokenizer.texts_to_sequences(X_train)
    tokenized_X_test = tokenizer.texts_to_sequences(X_test)
    MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
    tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding='post')
    tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding='post')
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=MAX_LENGTH))
    model.add(Flatten())
    model.add(Dense(64, activation="relu", kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4), activity_regularizer=L2(1e-4)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(tokenized_X_train, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=1)
    y_pred = model.predict(tokenized_X_test)
    y_pred = y_pred.argmax(axis=1)
    return classification_report(y_test, y_pred)


RNN = RNN()
FFNN = FFNN()
print(RNN)
print(FFNN)

# überaschenderweise ist das RNN (0.74) nur geringfügig besser als das FFNN (0.73)




