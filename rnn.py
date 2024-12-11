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

# splitting the dataset into 2 parts
# 20% testing
# 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
# below the names (x) are converted into lists, else
# keras would not know what to tokenize
X_train = X_train.to_list()
X_test = X_test.to_list()

# first adding the nationality (y) to the numbers
# somehow to_categorical from keras must be added
# else the system would derp around
y_train = to_number(y_train)
y_test = to_number(y_test)
y_train = to_categorical(y_train, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)

# Tokenizing the list input
tokenizer = Tokenizer(char_level=True)
# fitting the tokenizer on the "text"
tokenizer.fit_on_texts(X_train)
# length of the vocabulary (unique names?) in the training set
vocab_size = len(tokenizer.word_index)+1

# creating a sequence out of the text
tokenized_X_train = tokenizer.texts_to_sequences(X_train)
tokenized_X_test = tokenizer.texts_to_sequences(X_test)
# getting the maximum lenght of a name in the training data
MAX_LENGTH = max(len(tokenized_text) for tokenized_text in tokenized_X_train)
# padding the text
tokenized_X_train = pad_sequences(tokenized_X_train, maxlen=MAX_LENGTH, padding="post")
tokenized_X_test = pad_sequences(tokenized_X_test, maxlen=MAX_LENGTH, padding="post")

# regularizer = L1L2(l1=1e-5,l2=1e-4)
model = Sequential()
# Embeddings
model.add(Embedding(input_dim=vocab_size, output_dim=256))
# The next 3 lines of code are the simple recurrent neural network and an layer
# I do not understand how to add the regularizers
# => more work needed
model.add(SimpleRNN(256, activation="relu", dropout=0.3, recurrent_dropout=0.3,
                    return_sequences=True))
model.add(Bidirectional(SimpleRNN(256)))
# last dense layer with softmax activation function that is of size 6, the
# amount of end classes of the training set
model.add(Dense(6, activation="softmax"))

# compiling the model, fitting the model and printing a summary
model.compile(loss="crossentropy", optimizer=Adam(learning_rate=0.001))
model.fit(tokenized_X_train, y_train, epochs=100, verbose=1, batch_size=64)
model.summary()

# creating test indices, predicting and the printing an
# classification report of how good the model is
y_test_indices = y_test.argmax(axis=1)
y_pred = model.predict(tokenized_X_test)
y_pred = y_pred.argmax(axis=1)
print(classification_report(y_test_indices, y_pred))
