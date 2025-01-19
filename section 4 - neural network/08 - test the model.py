import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# Load the saved model and necessary components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the model architecture (same as training)
class MLPModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dims=[512, 256, 128], output_dim=1):
        super(MLPModel, self).__init__()

        # Embedding Layer with frozen weights
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True,
            padding_idx=0
        )

        # Calculate input dimension
        max_len = 1000  # Same as training
        input_dim = embedding_matrix.shape[1] * max_len

        # Create list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.5))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Combine all layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Get embeddings and flatten
        embedded = self.embedding(x)
        flattened = embedded.view(embedded.size(0), -1)

        # Forward pass through all layers
        return self.model(flattened)


# Dataset class (same as training)
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def load_and_test_model(test_file_path, train_file_path, model_path, word2vec_path):
    # Load Word2Vec embeddings
    print("Loading Word2Vec embeddings...")
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    # Create vocabulary (same as training)
    print("Creating vocabulary...")
    embedding_dim = 300
    vocab = {"<PAD>": 0, "<UNK>": 1}
    embedding_matrix = [np.zeros(embedding_dim), np.random.uniform(-0.01, 0.01, embedding_dim)]

    # Build vocabulary from training data first
    train_data = pd.read_csv(train_file_path)
    for text in train_data['text']:
        for word in text.split():
            if word not in vocab and word in word2vec:
                vocab[word] = len(vocab)
                embedding_matrix.append(word2vec[word])

    embedding_matrix = np.array(embedding_matrix)
    print(f"Vocabulary size: {len(vocab)}")

    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv(test_file_path)
    X_test = test_data['text']
    y_test = test_data['label']

    # Tokenize and convert text to sequences
    def text_to_sequence(text, vocab, max_len=1000):
        sequence = [vocab.get(word, vocab["<UNK>"]) for word in text.split()]
        if len(sequence) < max_len:
            sequence.extend([vocab["<PAD>"]] * (max_len - len(sequence)))
        return sequence[:max_len]

    print("Converting text to sequences...")
    X_test_seq = [text_to_sequence(text, vocab, 1000) for text in X_test]

    # Create test dataset and dataloader
    test_dataset = TextDataset(X_test_seq, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Load the model
    print("Loading the saved model...")
    model = MLPModel(embedding_matrix).to(device)

    # Load the state dict with safe loading
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Test the model
    print("Testing the model...")
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze(1)
            preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_preds)

    # Metrics for True News (class 1)
    precision_true = precision_score(test_labels, test_preds, pos_label=1)
    recall_true = recall_score(test_labels, test_preds, pos_label=1)
    f1_true = f1_score(test_labels, test_preds, pos_label=1)

    # Metrics for Fake News (class 0)
    precision_fake = precision_score(test_labels, test_preds, pos_label=0)
    recall_fake = recall_score(test_labels, test_preds, pos_label=0)
    f1_fake = f1_score(test_labels, test_preds, pos_label=0)

    # Full classification report
    report = classification_report(test_labels, test_preds)

    # Print results
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nMetrics for True News (class 1):")
    print(f"Precision (True News): {precision_true:.4f}")
    print(f"Recall (True News): {recall_true:.4f}")
    print(f"F1-Score (True News): {f1_true:.4f}")

    print("\nMetrics for Fake News (class 0):")
    print(f"Precision (Fake News): {precision_fake:.4f}")
    print(f"Recall (Fake News): {recall_fake:.4f}")
    print(f"F1-Score (Fake News): {f1_fake:.4f}")

    print("\nDetailed Classification Report:")
    print(report)

    # Return all metrics
    metrics = {
        'accuracy': accuracy,
        'precision_true': precision_true,
        'recall_true': recall_true,
        'f1_true': f1_true,
        'precision_fake': precision_fake,
        'recall_fake': recall_fake,
        'f1_fake': f1_fake,
        'full_report': report
    }

    return metrics


if __name__ == "__main__":
    # Specify your paths
    test_file_path = "test.csv"
    train_file_path = "train.csv"  # Added train file path
    model_path = "best_mlp_model.pth"
    word2vec_path = "../GoogleNews-vectors-negative300.bin.gz"

    metrics = load_and_test_model(test_file_path, train_file_path, model_path, word2vec_path)