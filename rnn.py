import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.utils import to_categorical
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41, stratify=y)
X_train = X_train.to_list()
X_test = X_test.to_list()

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index)+1
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

y_train = to_number(y_train)
y_test = to_number(y_test)
y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)

regularizer = L1L2(l1=1e-5, l2=1e-4)
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=MAX_LENGTH))
model.add(SimpleRNN(64, activation="relu", dropout=0.3, recurrent_dropout=0.3, 
                    return_sequences=True, kernel_regularizer=regularizer, 
                    bias_regularizer=regularizer, activity_regularizer=regularizer, 
                    recurrent_regularizer=regularizer))
model.add(Bidirectional(SimpleRNN(64)))
model.add(Dense(6, activation="softmax"))
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=Adam(learning_rate=0.001))
model.summary()
model.fit(tokenized_X_train, y_train, validation_split=0.1, epochs=40, verbose=1, batch_size=20)


y_test_indices = y_test.argmax(axis=1)
y_pred = model.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test_indices, y_pred))