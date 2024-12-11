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
number_classes = len(df.nationality.unique())
y = to_number(df.nationality)
print(number_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1337, stratify=y)
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
# for checking out some of the word sequences and "retranslating" them:
# print(tokenizer.word_index)
# for test in tokenized_X_test:
#     print(test)


# FFNN macro avg F1: 0.59 weighted avg: 0.62: --> Actually best result with current settings
model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(vocab_size, 150, input_length=MAX_LENGTH))
model.add(Flatten())
# Added Regularizers
model.add(Dense(100, activation="relu", kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4),
                activity_regularizer=L2(1e-5)))
model.add(Dense(100, activation="relu", kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L1(1e-4),
                activity_regularizer=L2(1e-5)))
model.add(Dense(number_classes, activation="softmax"))
# Changed optimizer to Adam
model.compile(loss="crossentropy", optimizer=Adam(learning_rate=0.01))
model.summary()

# UNIDIRECTIONAL RNN: macro avg F1: 0.57 weighted avg: 0.60 --> Worst with current settings somehow
# model = Sequential()
# model.add(Input(shape=(MAX_LENGTH,)))
# model.add(Embedding(vocab_size, 150, input_length=MAX_LENGTH))
# model.add(SimpleRNN(64, activation="relu", dropout=0.3, recurrent_dropout=0.3,
#                     return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4),
#                     activity_regularizer=L1(1e-5), recurrent_regularizer=L1L2(l1=1e-5, l2=1e-4)))
# model.add(Flatten())
# model.add(Dense(number_classes, activation="softmax"))
# model.compile(loss="crossentropy", optimizer=Adam(learning_rate=0.001))
# model.summary()

# BIDIRECTIONAL RNN: macro avg F1: 0.58 weighted avg: 0.60 --> Slightly better than unidir
# model = Sequential()
# model.add(Input(shape=(MAX_LENGTH,)))
# model.add(Embedding(vocab_size, 150, input_length=MAX_LENGTH))
# model.add(SimpleRNN(64, activation="relu", dropout=0.3, recurrent_dropout=0.3,
#                     return_sequences=True, kernel_regularizer=L1L2(l1=1e-5, l2=1e-4), bias_regularizer=L2(1e-4),
#                     activity_regularizer=L1(1e-5), recurrent_regularizer=L1L2(l1=1e-5, l2=1e-4)))
# model.add(Bidirectional(SimpleRNN(64)))
# model.add(Flatten())
# model.add(Dense(number_classes, activation="softmax"))
# model.compile(loss="crossentropy", optimizer=Adam(learning_rate=0.001))
# model.summary()

# without that numpy conversion I always get an error and the model isn't working at all:
tokenized_X_train = np.array(tokenized_X_train)
y_train = np.array(y_train)
model.fit(tokenized_X_train, y_train, epochs=5, verbose=1)
y_pred = model.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))


