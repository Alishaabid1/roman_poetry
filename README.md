Roman Urdu Poetry Generation Using RNN
This project demonstrates how to generate Roman Urdu poetry using an RNN (Recurrent Neural Network) model, specifically built with TensorFlow and Keras. We will train an RNN-based model using a dataset of Roman Urdu poetry and deploy it with a simple front-end using Gradio.

Table of Contents
Introduction
Dataset
Model
Training
Generating Poetry
Deployment with Gradio
Setup Instructions
Contributing
License
Introduction
This project is focused on training an RNN-based model to generate Roman Urdu poetry. The model learns the structure and style of the poetry by looking at various famous poets' works in Roman Urdu. Once trained, the model can be used to generate new lines of poetry.

Additionally, a Gradio frontend is created to make it easy to interact with the model and generate poetry on demand.

Dataset
The dataset for training consists of Roman Urdu poetry collected from various sources. The dataset includes columns for the poet’s name and the poetry text itself.

Dataset Columns
ID: Unique identifier for each poem.
Poet: The poet’s name.
Poetry: The Roman Urdu poetry text.
Sample:

ID	Poet	Poetry
1	Ahmad Faraz	tarīq-e-ishq meñ mujh ko koī kāmil nahīñ miltā ga.e farhād o majnūñ ab kisī se dil nahīñ miltā
2	Allama Iqbal	ishq hi zindagi hai, ishq hi mazhab hai, kisī se bhi agar kuchh mangna ho toh ishq hi dikhāye
Model
For this task, we use a SimpleRNN model from Keras with the following architecture:

Embedding Layer: To convert words into embeddings.
Simple RNN Layer: To capture the sequential nature of poetry.
Dropout Layer: To avoid overfitting.
Dense Layer: To predict the next word in the sequence.
The model is trained on sequences of words from the poetry text, and it generates the next word given a sequence of previous words.

Hyperparameters:
Sequence length: 5 (i.e., the model looks at the last 5 words to predict the next word)
Vocabulary size: The total number of unique words in the dataset.
Hidden units in RNN: 128 units in the RNN layer.
Training
The model is trained on the tokenized version of the poetry dataset, where each word is converted to an integer based on a word-to-index mapping. After training, the model is capable of generating poetry by predicting the next word in a sequence.

Steps:
Data Preprocessing: Tokenize the poetry, create sequences of words, and prepare the input-output pairs.
Model Training: Train the RNN model on the sequences using sparse categorical cross-entropy loss and Adam optimizer.
Model Evaluation: Evaluate the performance of the model on unseen sequences.
Generating Poetry
Once the model is trained, we can generate Roman Urdu poetry by feeding it an initial seed sequence. The model predicts one word at a time and appends it to the sequence, generating a new line of poetry.

python
Copy
Edit
def generate_poetry(seed_text, model, idx2word, word2idx, max_sequence_length=5):
    for _ in range(50):  # Generate up to 50 words
        tokenized_seed = [word2idx[word] for word in seed_text.split()]
        tokenized_seed = pad_sequences([tokenized_seed], maxlen=max_sequence_length, padding='post')
        predicted_word_idx = model.predict(tokenized_seed, verbose=0).argmax(axis=1)[0]
        predicted_word = idx2word[predicted_word_idx]
        seed_text += ' ' + predicted_word
        if predicted_word == '<END>':
            break
    return seed_text
Deployment with Gradio
To make the poetry generation accessible to everyone, we use Gradio to deploy the model in an interactive web interface. The user can enter a seed word or phrase, and the model will generate the rest of the poetry.

python
Copy
Edit
import gradio as gr

def poetry_generator(seed_text):
    generated_poetry = generate_poetry(seed_text, model_rnn, idx2word, word2idx)
    return generated_poetry

iface = gr.Interface(fn=poetry_generator, inputs="text", outputs="text", live=True)
iface.launch()
Steps:
Gradio Setup: Define a simple function that takes input from the user, generates poetry, and returns it.
Launch: Use the launch() function to start the Gradio interface.
Setup Instructions
To run this project locally, follow these steps:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/roman-urdu-poetry-generation.git
cd roman-urdu-poetry-generation
Install dependencies:

Install the required libraries using pip.

bash
Copy
Edit
pip install -r requirements.txt
Download the dataset:

Make sure the dataset (Roman-Urdu-Poetry.csv) is placed in the root directory of the project.

Run the notebook:

If you are using Google Colab, you can upload your files and execute the notebook for training. For local environments, use Jupyter.

Deploy Gradio Interface:

Launch the poetry generation interface using Gradio:

bash
Copy
Edit
python app.py
This will open a web interface in your browser for poetry generation.

Contributing
Feel free to fork this project, open issues, or submit pull requests. Contributions are welcome!

Guidelines:
Add more poetry datasets to improve the diversity of the model.
Tweak the model architecture for improved results.
Add additional features, such as saving generated poetry or multilingual support.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Blog Outline
Introduction:

The motivation for the project: generating poetry using AI.
Brief overview of the dataset and model used.
Dataset:

Discuss the Roman Urdu poetry dataset and its structure.
Explanation of how data is preprocessed (tokenization, padding, etc.).
Model Training:

Explanation of the RNN architecture.
Model's training process, including how the loss function and optimizer were chosen.
Poetry Generation:

Walkthrough of how the model generates poetry.
Challenges faced and how they were overcome.
Deployment:

How the poetry generation model is deployed with Gradio.
User instructions for interacting with the model.
Conclusion:

Future improvements for the project.
Possible applications of AI in poetry generation.
