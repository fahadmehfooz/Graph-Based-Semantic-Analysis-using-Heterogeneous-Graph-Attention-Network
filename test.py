import pandas as pd
import re
import spacy
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import seaborn as sns
import torch.nn.functional as F
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk import PorterStemmer, WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix

import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore")

'''

'''

'''
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

data= pd.read_csv('train-1.csv')
data.head()

data['label']=data['label'].astype(int)
data.head()

import re
import spacy
nlp = spacy.load('en_core_web_sm')

# Function for text preprocessing and lemmatization
def preprocess_and_lemmatize(text):
    text = text.lower()
    text = re.sub(r'\?+', '?', text)
    text = re.sub(r'\!+', '!', text)
    text = re.sub(r'\/+', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'-lrb-', '-', text)
    text = re.sub(r'-rrb-', '-', text)
    text = re.sub(r'\-+', '-', text)
    # text = re.sub(r"\b(i['`’]ve|i['`’]m|i['`’]ll)\b", lambda m: m.group(0).replace("'", " "), text)
    text = re.sub(
    r"\b(i['’]ve|i['’]m|i['’]ll)\b",
    lambda m: {
        "i've": "I have",
        "i’m": "I am",
        "i'm": "I am",
        "i’ll": "I will",
        "i'll": "I will"
    }[m.group(0).lower()],
    text,
    flags=re.IGNORECASE
)
    text = re.sub(r" - ", " . ", text)

    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    lemmatized_text = ' '.join(lemmatized_tokens)
    lemmatized_text = re.sub(r"'s\b", "s", lemmatized_text)
    return lemmatized_text

data['lemmatized_text'] = data['text'].apply(preprocess_and_lemmatize)
data.head()

"""# Visualizing WordClouds according to Metaphor IDs (on RAW data)"""

# Lets plot a word cloud of all the text according to metaphorID

# Import necessary libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Group the data by metaphorID
grouped_data = data.groupby('metaphorID')['text'].apply(lambda x: ' '.join(x))

# Loop through each metaphorID and generate a word cloud
for metaphor_id, text in grouped_data.items():
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the generated image:
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for Metaphor ID: {metaphor_id}")
    plt.show()

metaphor_keywords = {
    0: 'road',
    1: 'candle',
    2: 'light',
    3: 'spice',
    4: 'ride',
    5: 'train',
    6: 'boat'
}

# Function to extract metaphor-related sentences
def extract_metaphor_sentences(df):
    imp_text = []

    for _, row in df.iterrows():
        metaphor_keyword = metaphor_keywords.get(row['metaphorID'], '')
        selected_sentence = ""

        for sentence in row['lemmatized_text'].split('.'):
            if metaphor_keyword in sentence:
                # Break condition: prioritize sentences without specific punctuation
                if not any(punct in sentence for punct in [' - ', ' , ', ' ; ', ' ? ', ' ! ']):
                    selected_sentence = sentence
                    break

                # Secondary prioritization based on punctuation
                for punct in [' , ', ' ; ', ' ? ', ' ! ', ' - ']:
                    if punct in sentence:
                        for sub_sentence in sentence.split(punct):
                            if metaphor_keyword in sub_sentence:
                                selected_sentence = sub_sentence
                                break
                        if selected_sentence:
                            break

        if selected_sentence:
            imp_text.append(selected_sentence)

    return imp_text

def final_preprocess_data(df):

    # Preprocess and lemmatize text
    df['lemmatized_text'] = df['text'].apply(preprocess_and_lemmatize)

    # Extract metaphor-related sentences
    imp_text = extract_metaphor_sentences(df)

    # Create cleaned DataFrame with important metaphor-related sentences
    cleaned_df = pd.DataFrame({'text': imp_text})
    cleaned_df[["metaphorID", "label"]] = df.iloc[:, :2]
    print(cleaned_df.head())
    # Map labels to target binary values
    cleaned_df["target"] = cleaned_df["label"].map({True: 1, False: 0})
    cleaned_df = cleaned_df.drop("label", axis=1)

    # Map metaphorID to metaphor keywords
    cleaned_df["metaphor"] = cleaned_df["metaphorID"].map(metaphor_keywords)
    cleaned_df = cleaned_df.drop("metaphorID", axis=1)

    return cleaned_df

cleaned_df = final_preprocess_data(data)

cleaned_df.head()

"""# Visualizing WordClouds according to Metaphor IDs (on CLEAN data)"""

# Group the data by metaphorID
grouped_data_2 = cleaned_df.groupby('metaphor')['text'].apply(lambda x: ' '.join(x))

# Loop through each metaphorID and generate a word cloud
for metaphor, text in grouped_data_2.items():
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the generated image:
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud for Metaphor ID(post cleaning):  {metaphor}")
    plt.show()

X, y = cleaned_df.drop("target", axis = 1), data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

X_train

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Masking, SimpleRNN, StackedRNNCells, GRU
from tensorflow.keras.optimizers import Adam

vocab_size = 3500
embedding_dim = 16
max_length = 300 # almost all the texts are within 300 words
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train['text'])

word_index = tokenizer.word_index

word_index

training_seq = tokenizer.texts_to_sequences(X_train['text'])
training_padded = pad_sequences(training_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_seq = tokenizer.texts_to_sequences(X_test['text'])
testing_padded = pad_sequences(testing_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_padded

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Masking, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split

# Define the model with additional regularization and dropout layers
model_LSTM = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=300),
    Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))),
    Dropout(0.3),  # Dropout to reduce overfitting
    Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with learning rate
optimizer = Adam(learning_rate=0.0005)

model_LSTM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define early stopping and model checkpoint callbacks to monitor 'val_loss'
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Convert data to numpy arrays
training_padded = np.array(training_padded)
training_labels = np.array(y_train)
testing_padded = np.array(testing_padded)
testing_labels = np.array(y_test)

# Split training data to create a separate validation set
training_padded, validation_padded, training_labels, validation_labels = train_test_split(
    training_padded, training_labels, test_size=0.2, random_state=42
)

# Fit the model using the training and validation sets
num_epochs = 150
history = model_LSTM.fit(
    training_padded, training_labels,
    epochs=num_epochs,
    validation_data=(validation_padded, validation_labels),
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

import matplotlib.pyplot as plt

# Assuming 'history' is the history object returned by model.fit()

# Create a figure with subplots
plt.figure(figsize=(10, 5))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle='-', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--', marker='x')
plt.title('Train and Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linestyle='-', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', marker='x')
plt.title('Train and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

y_pred=np.round(model_LSTM.predict(testing_padded))
y_pred=y_pred.flatten()
y_pred

from sklearn.metrics import classification_report
# Generate classification report
report = classification_report( testing_labels,y_pred)

# Print the classification report
print("Classification Report:\n", report)
'''


def test(test_data, model_state_path):

    nltk.download('stopwords')
    nltk.download('wordnet')

    # Function for text preprocessing and lemmatization
    def preprocess_and_lemmatize(text):
        text = text.lower()
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'\!+', '!', text)
        text = re.sub(r'\/+', '', text)
        text = re.sub(r'`', '', text)
        text = re.sub(r'-lrb-', '-', text)
        text = re.sub(r'-rrb-', '-', text)
        text = re.sub(r'\-+', '-', text)
        text = re.sub(r"\b(i['`’]ve|i['`’]m|i['`’]ll)\b",
                      lambda m: m.group(0).replace("'", " "), text)
        text = re.sub(r" - ", " . ", text)

        doc = nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        lemmatized_text = ' '.join(lemmatized_tokens)
        lemmatized_text = re.sub(r"'s\b", "s", lemmatized_text)
        return lemmatized_text

    # Function to extract metaphor-related sentences
    def extract_metaphor_sentences(df):
        metaphor_keywords = {
            0: 'road',
            1: 'candle',
            2: 'light',
            3: 'spice',
            4: 'ride',
            5: 'train',
            6: 'boat'
        }

        imp_text = []

        for _, row in df.iterrows():
            metaphor_keyword = metaphor_keywords.get(row['metaphorID'], '')
            selected_sentence = ""

            for sentence in row['lemmatized_text'].split('.'):
                if metaphor_keyword in sentence:
                    # Break condition: prioritize sentences without specific punctuation
                    if not any(punct in sentence for punct in [' - ', ' , ', ' ; ', ' ? ', ' ! ']):
                        selected_sentence = sentence
                        break

                    # Secondary prioritization based on punctuation
                    for punct in [' , ', ' ; ', ' ? ', ' ! ', ' - ']:
                        if punct in sentence:
                            for sub_sentence in sentence.split(punct):
                                if metaphor_keyword in sub_sentence:
                                    selected_sentence = sub_sentence
                                    break
                            if selected_sentence:
                                break

            if selected_sentence:
                imp_text.append(selected_sentence)

        return imp_text

    def clean_and_process_data(df):
        metaphor_keywords = {
            0: 'road',
            1: 'candle',
            2: 'light',
            3: 'spice',
            4: 'ride',
            5: 'train',
            6: 'boat'
        }
        df['lemmatized_text'] = df['text'].apply(preprocess_and_lemmatize)

        # Extract metaphor-related sentences
        imp_text = extract_metaphor_sentences(df)

        # Create cleaned DataFrame with important metaphor-related sentences
        cleaned_df = pd.DataFrame({'text': imp_text})
        cleaned_df[["metaphorID", "label"]] = df.iloc[:, :2].values

        # Map labels to target binary values
        cleaned_df["label"] = cleaned_df["label"].map({True: 1, False: 0})

        # Map metaphorID to metaphor keywords
        cleaned_df["metaphor"] = cleaned_df["metaphorID"].map(
            metaphor_keywords)
        cleaned_df = cleaned_df.drop("metaphorID", axis=1)

        return cleaned_df

    def create_bert_embeddings(text_series):
        embeddings = []
        for text in text_series:
            inputs = tokenizer(text, return_tensors="pt",
                               padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(
                dim=1).squeeze().numpy())
        return torch.tensor(np.array(embeddings), dtype=torch.float)

    # Create a heterogeneous graph
    def create_hetero_graph(texts, metas):
        G = nx.Graph()
        word_to_doc_edges = []
        unique_words = set()

        for doc_id, (text, meta) in enumerate(zip(texts, metas)):
            words = set(str(text).lower().split() + str(meta).lower().split())
            unique_words.update(words)
            for word in words:
                word_to_doc_edges.append(('word', word, 'doc', doc_id))

        return list(unique_words), word_to_doc_edges

    class HeteroGNN(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels, num_layers, heads=4):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('word', 'to', 'doc'): GATConv((-1, -1), hidden_channels // heads, heads=heads, concat=True, add_self_loops=False),
                    ('doc', 'to', 'word'): GATConv((-1, -1), hidden_channels // heads, heads=heads, concat=True, add_self_loops=False),
                })
                self.convs.append(conv)
            self.lin = torch.nn.Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
            logits = self.lin(x_dict['doc'])
            return F.softmax(logits, dim=1)

    # Training and evaluation functions
    def train(model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(out, data['doc'].y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(model, data):

        model.eval()

        with torch.no_grad():
            # We are not leaking any data both x_dict, edge_index_dict both do not have any y values
            out = model(data.x_dict, data.edge_index_dict)
            pred = out.argmax(dim=1)
            return pred

    def compute_metrics(train_true, train_pred, train_probs, test_true, test_pred, test_probs):
        def calculate_metrics(y_true, y_pred, y_probs):
            metrics = {
                "Precision": precision_score(y_true, y_pred, average="binary"),
                "Recall": recall_score(y_true, y_pred, average="binary"),
                "F1-Score": f1_score(y_true, y_pred, average="binary"),
                "Accuracy": accuracy_score(y_true, y_pred),
                "ROC-AUC": roc_auc_score(y_true, y_probs[:, 1]) if y_probs is not None else None
            }
            return metrics

        train_metrics = calculate_metrics(train_true, train_pred, train_probs)
        test_metrics = calculate_metrics(test_true, test_pred, test_probs)

        df = pd.DataFrame({
            'Train': train_metrics,
            'Test': test_metrics
        }).transpose()

        print(df)

        print("\nClassification Report (Train):")
        print(classification_report(train_true, train_pred))

        print("\nClassification Report (Test):")
        print(classification_report(test_true, test_pred))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # ROC-AUC plot
        for true, probs, name in [(train_true, train_probs, 'Train'), (test_true, test_probs, 'Test')]:
            fpr, tpr, _ = roc_curve(true, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")

        # Precision-Recall plot
        for true, probs, name in [(train_true, train_probs, 'Train'), (test_true, test_probs, 'Test')]:
            precision, recall, _ = precision_recall_curve(true, probs[:, 1])
            ax2.plot(recall, precision, label=f'{name}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()

        # Confusion Matrix for Train data
        train_cm = confusion_matrix(train_true, train_pred)
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix (Train)')

        # Confusion Matrix for Test data
        test_cm = confusion_matrix(test_true, test_pred)
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title('Confusion Matrix (Test)')

        plt.tight_layout()
        plt.show()

        return df

    def compute_metrics(y_true, y_pred):
        metrics = {
            "Precision": precision_score(y_true, y_pred, average="macro"),
            "Recall": recall_score(y_true, y_pred, average="macro"),
            "F1-Score": f1_score(y_true, y_pred, average="macro"),
            "Accuracy": accuracy_score(y_true, y_pred),
        }

        return metrics

    nlp = spacy.load("en_core_web_sm")

    cleaned_dummy = clean_and_process_data(test_data)
    dummy_text, dummy_meta = cleaned_dummy["text"], cleaned_dummy["metaphor"]

    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    dummy_words, dummy_edges = create_hetero_graph(dummy_text, dummy_meta)
    dummy_word_features = create_bert_embeddings(dummy_words)
    dummy_doc_features = create_bert_embeddings(
        [text + " " + meta for text, meta in zip(dummy_text, dummy_meta)])

    dummy_data = HeteroData()
    dummy_data['word'].x = dummy_word_features
    dummy_data['doc'].x = dummy_doc_features
    dummy_data['word', 'to', 'doc'].edge_index = torch.tensor([[dummy_words.index(
        edge[1]) for edge in dummy_edges], [edge[3] for edge in dummy_edges]], dtype=torch.long)
    dummy_data['doc'].y = torch.tensor(cleaned_dummy["label"])

    # Add reverse edges
    dummy_data['doc', 'to', 'word'].edge_index = dummy_data['word',
                                                            'to', 'doc'].edge_index.flip([0])

    best_params = {'hidden_channels': 32,
                   'num_layers': 2, 'heads': 8, 'lr': 0.001}

    final_model = HeteroGNN(
        hidden_channels=best_params['hidden_channels'],
        out_channels=2,
        num_layers=best_params['num_layers'],
        heads=best_params['heads']
    )

    final_model.load_state_dict(torch.load(model_state_path))

    final_model.eval()

    predicted = evaluate(final_model, dummy_data)

    results = compute_metrics(dummy_data['doc'].y, predicted)
    results_df = pd.DataFrame(data=[results.values()], columns=results.keys())

    print("Results on test data:")
    print(results_df)
    print()

    test_data["predicted"] = predicted
    test_data["predicted"] = test_data["predicted"].map({1: True, 0: False})

    test_data.to_csv("predictions.csv")

    return test_data


def main():
    file_name = input("Enter testing dataset name: ").strip()

    # Check if the input is empty
    if not file_name:
        print("Error: File name cannot be empty. Please enter a valid file name.")
        return

    # Load the test data
    try:
        test_data = pd.read_csv(file_name)
    except FileNotFoundError:
        print(
            f"Error: File '{file_name}' not found. Please check the filename and try again.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Path to the model state dictionary
    model_state_path = 'GNN_optimal_state_dict.pth'

    # Assuming 'test' is a function defined elsewhere to process the data
    predictions = test(test_data, model_state_path)

    # Print or save the predictions
    print("Predictions generated successfully!")
    print(predictions)


if __name__ == "__main__":
    main()
