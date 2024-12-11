import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the CSV file
df = pd.read_csv("names.csv", sep=",")

# Print column names to see what we're working with
print("Column names:", df.columns)

# Assuming the columns are named differently, let's use the first two columns
# Adjust these column names based on what you see in the print output
name_column = df.columns[0]
origin_column = df.columns[1]

# Function to convert labels to numbers
def to_number(labels):
    unique_labels = list(set(labels))
    return [unique_labels.index(label) for label in labels]

# Prepare the data
X = df[name_column].values
y = to_number(df[origin_column].values)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize characters
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)

# Convert to sequences and pad
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max(len(seq) for seq in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Convert to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

vocab_size = len(tokenizer.word_index) + 1

# Get the number of unique classes
num_classes = len(set(y))

# Define and train a simple RNN model
model = Sequential([
    Embedding(vocab_size, 32, input_length=max_length),
    SimpleRNN(64),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_pad, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes))

# results:[] precision    recall  f1-score   support

#            0       0.84      0.79      0.81        58
#            1       0.71      0.53      0.61        51
#            2       0.75      0.67      0.71        36
#            3       0.65      0.73      0.69        64
#            4       0.60      0.66      0.63        41
#            5       0.39      0.52      0.45        25

#     accuracy                           0.67       275
#    macro avg       0.66      0.65      0.65       275
# weighted avg       0.68      0.67      0.67       275

