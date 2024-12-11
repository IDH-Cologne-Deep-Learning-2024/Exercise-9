import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, SimpleRNN, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Function to convert labels to numbers
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

# Load dataset
df = pd.read_csv("names.csv")

# Preprocess data
labels = df['nationality']
names = df['name']
number_labels = to_number(labels)
class_names = ['Dutch', 'Thai', 'Russian', 'Chinese', 'Hawaiian', 'Hungarian']  # Class name list

# Tokenize names at the character level
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(names)
sequences = tokenizer.texts_to_sequences(names)

# Pad sequences
max_length = max(len(name) for name in names)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, number_labels, test_size=0.2, random_state=42
)

# Convert labels to one-hot encoding
num_classes = len(set(number_labels))
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Build RNN model
def build_rnn_model(input_dim, output_dim, max_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length),
        SimpleRNN(64, activation='relu', kernel_regularizer=L2(0.01)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build BiRNN model
def build_birnn_model(input_dim, output_dim, max_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length),
        Bidirectional(SimpleRNN(64, activation='relu', kernel_regularizer=L2(0.01))),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build FFNN model
def build_ffnn_model(input_dim, output_dim, max_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=max_length),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=L2(0.01)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training and evaluation function
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Accuracy: {accuracy}")
    return model

# Tokenizer vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Train and evaluate RNN
print("Training RNN...")
rnn_model = build_rnn_model(vocab_size, 32, max_length)
train_and_evaluate_model(rnn_model, X_train, y_train, X_test, y_test)

# Train and evaluate BiRNN
print("Training BiRNN...")
birnn_model = build_birnn_model(vocab_size, 32, max_length)
train_and_evaluate_model(birnn_model, X_train, y_train, X_test, y_test)

# Train and evaluate FFNN
print("Training FFNN...")
ffnn_model = build_ffnn_model(vocab_size, 32, max_length)
train_and_evaluate_model(ffnn_model, X_train, y_train, X_test, y_test)