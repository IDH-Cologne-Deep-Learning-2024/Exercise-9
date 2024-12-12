import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
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


tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['name'])  
X = tokenizer.texts_to_sequences(df['name'])
X = pad_sequences(X)  

y = pd.get_dummies(df['nationality']).values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_rnn = Sequential()
model_rnn.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=X_train.shape[1]))  
model_rnn.add(SimpleRNN(64, return_sequences=True))  
model_rnn.add(Dropout(0.5)) 
model_rnn.add(SimpleRNN(64))  
model_rnn.add(Dense(y_train.shape[1], activation='softmax'))  

model_rnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model_rnn.summary()


history_rnn = model_rnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


model_birnn = Sequential()
model_birnn.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=X_train.shape[1]))
model_birnn.add(Bidirectional(SimpleRNN(64, return_sequences=True)))  
model_birnn.add(Dropout(0.5))  
model_birnn.add(Bidirectional(SimpleRNN(64)))  
model_birnn.add(Dense(y_train.shape[1], activation='softmax'))  

model_birnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model_birnn.summary()


history_birnn = model_birnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


model_ffnn = Sequential()
model_ffnn.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=X_train.shape[1]))
model_ffnn.add(Flatten())  
model_ffnn.add(Dense(128, activation='relu'))  
model_ffnn.add(Dropout(0.5))  
model_ffnn.add(Dense(y_train.shape[1], activation='softmax')) 

model_ffnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model_ffnn.summary()


history_ffnn = model_ffnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)


y_pred_rnn = model_rnn.predict(X_test)
y_pred_birnn = model_birnn.predict(X_test)
y_pred_ffnn = model_ffnn.predict(X_test)


print("RNN Classification Report:")
print(classification_report(y_test.argmax(axis=1), y_pred_rnn.argmax(axis=1)))

print("BiRNN Classification Report:")
print(classification_report(y_test.argmax(axis=1), y_pred_birnn.argmax(axis=1)))

print("FFNN Classification Report:")
print(classification_report(y_test.argmax(axis=1), y_pred_ffnn.argmax(axis=1)))