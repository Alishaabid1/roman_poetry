# Roman Urdu Poetry Generator

This project demonstrates how to generate **Roman Urdu poetry** using a Recurrent Neural Network (RNN) model built with TensorFlow and Keras. The model is trained on a dataset of Roman Urdu poetry, and once trained, it can sample new poetry from a given seed phrase. The generated text is then translated into Urdu script using the Deep Translator. Finally, a simple front-end is deployed using Gradio so that anyone can interact with the model.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model and Training](#model-and-training)
4. [Generating and Translating Poetry](#generating-and-translating-poetry)
5. [Deployment with Gradio](#deployment-with-gradio)
6. [Setup Instructions](#setup-instructions)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction

The goal of this project is to train an RNN (or its variant, such as LSTM or GRU) on Roman Urdu poetry so that we can later sample new, creative poetry. After training, the model generates poetry based on a seed text in Roman Urdu, and then the output is translated into Urdu script. The project also includes a simple front-end built with Gradio for quick interaction and demonstration.

---

## Dataset

The dataset used in this project is a CSV file named `Roman-Urdu-Poetry.csv`, which contains the following columns:

- **ID**: Unique identifier for each poem.
- **Poet**: Name of the poet.
- **Poetry**: The Roman Urdu poetry text.

A sample row from the dataset:

| ID  | Poet         | Poetry                                                                                       |
|-----|--------------|----------------------------------------------------------------------------------------------|
| 1   | Ahmad Faraz  | tarīq-e-ishq meñ mujh ko koī kāmil nahīñ miltā ga.e farhād o majnūñ ab kisī se dil nahīñ miltā |

---

## Model and Training

The model is built using a simple RNN architecture in Keras. The steps include:

1. **Data Preprocessing**:  
   - Tokenize the poetry text.
   - Convert the text into sequences of integers.
   - Create input-output pairs for training.
   - Reshape the input for the RNN (samples, time_steps, features).

2. **Model Architecture**:  
   - Two SimpleRNN layers (with dropout for regularization).
   - A Dense output layer with softmax activation to predict the next word.

3. **Training**:  
   - The model is compiled with the `sparse_categorical_crossentropy` loss function and the `adam` optimizer.
   - After training, the model is saved to a file for later use.

### Training Code

```python
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK data
nltk.download('punkt')

# Load the poetry dataset (Kaggle path)
df = pd.read_csv('/kaggle/input/roman-urdu-poetry-csv/Roman-Urdu-Poetry.csv')

# Check column names and standardize (rename "Poetry" to "poetry")
print("Dataset Columns:", df.columns)
df.rename(columns={"Poetry": "poetry"}, inplace=True)

# Drop missing values and convert poetry to list of strings
df = df.dropna(subset=['poetry'])
poetry_texts = df['poetry'].astype(str).tolist()

# Create a tokenizer and fit on the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poetry_texts)

# Convert poetry texts to sequences of integers
sequences = tokenizer.texts_to_sequences(poetry_texts)

# Prepare Input (X) and Output (y) sequences
sequence_length = 5  # You can adjust this as needed
X_rnn = []
y_rnn = []
for seq in sequences:
    for i in range(len(seq) - sequence_length):
        X_rnn.append(seq[i:i + sequence_length])  # Input sequence
        y_rnn.append(seq[i + sequence_length])      # Next word as output

X_rnn = np.array(X_rnn)
y_rnn = np.array(y_rnn)

# Reshape X_rnn for RNN: (samples, time_steps, features)
X_rnn = X_rnn.reshape(X_rnn.shape[0], X_rnn.shape[1], 1)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)

# Define the RNN Model using SimpleRNN
model_rnn = Sequential([
    SimpleRNN(128, return_sequences=True, input_shape=(X_rnn.shape[1], 1)),
    Dropout(0.2),
    SimpleRNN(128),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model_rnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_rnn.fit(X_rnn, y_rnn, epochs=5, batch_size=64)

# Save the trained model
model_rnn.save("/kaggle/working/rnn_poetry_model.h5")
print("Model saved successfully!")
