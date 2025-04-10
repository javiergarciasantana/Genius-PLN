import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
import random
from sklearn.utils import shuffle


# ----------------------------- Vocabulary Management -----------------------------
class Vocabulary:
    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.size = 0

    def add_token(self, token):
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.size
            self.idx_to_token[self.size] = token
            self.size += 1

    def encode(self, tokens):
        return [self.token_to_idx[token] for token in tokens]

    def decode(self, indices):
        return [self.idx_to_token[idx] for idx in indices]


# ----------------------------- Model Definition (Elman RNN) -----------------------------
class ElmanRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim):
        super(ElmanRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output 1 for binary classification

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: [batch_size, sequence_length, embed_size]
        output, hidden = self.rnn(
            embedded
        )  # Shape: [batch_size, sequence_length, hidden_dim]
        logits = self.fc(hidden[-1])  # Use the final hidden state
        return logits


# ----------------------------- Dataset Handling -----------------------------
def read_and_tokenize(filename):
    """
    Reads a file and creates a list where each element is a list of words in a sentence.
    Args:
        filename (str): Path to the input file.
    Returns:
        list: A list of sentences, where each sentence is a list of words.
    """
    sentences = []
    with open(filename, "r", encoding="utf-8-sig") as f:
        content = f.read()
        raw_sentences = content.split("###")  # Split by double newlines
        for raw_sentence in raw_sentences:
            tokens = raw_sentence.replace("\n", " ").strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, vocab, context_size):
        self.vocab = vocab
        self.context_size = context_size
        self.data = []
        self.pad_idx = vocab.token_to_idx["<PAD>"]
        for sentence, label in zip(sentences, labels):
            encoded_sentence = vocab.encode(sentence)
            if len(encoded_sentence) < context_size:
                encoded_sentence += [self.pad_idx] * (
                    context_size - len(encoded_sentence)
                )
            else:
                encoded_sentence = encoded_sentence[:context_size]
            self.data.append((encoded_sentence, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, label = self.data[idx]
        return torch.tensor(inputs), torch.tensor(label)


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return inputs, labels


# ----------------------------- Training Function -----------------------------
def train_model(model, data_loader, num_epochs, learning_rate, device="cpu"):
    model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_batches = len(data_loader)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)  # Shape: [batch_size]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}"
                )
        avg_loss = total_loss / total_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")


# ----------------------------- Evaluation Function -----------------------------
def evaluate_model(model, data_loader, device="cpu"):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(-1)  # Shape: [batch_size]
            predictions = (torch.sigmoid(outputs) > 0.5).long()  # Threshold at 0.5
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


# ----------------------------- Main Script -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Read and tokenize sentences from two files
    file1 = (
        r"G:\Mi unidad\24-25\docencia\iaa\practica\alltoghether\content\all_lyrics1.txt"
    )
    file2 = (
        r"G:\Mi unidad\24-25\docencia\iaa\practica\alltoghether\content\all_lyrics2.txt"
    )
    sentences_class0 = read_and_tokenize(file1)
    sentences_class1 = read_and_tokenize(file2)

    sentences_class0 = sentences_class0[0:30]
    sentences_class0 = sentences_class1[0:30]

    # Step 2: Build vocabulary
    vocab = Vocabulary()
    for sentence in sentences_class0 + sentences_class1:
        for token in sentence:
            vocab.add_token(token)
    vocab.add_token("<PAD>")  # Add padding token

    # Step 3: Assign labels (0 for class 0, 1 for class 1)
    labels_class0 = [0] * len(sentences_class0)
    labels_class1 = [1] * len(sentences_class1)

    # Combine sentences and labels
    sentences = sentences_class0 + sentences_class1
    labels = labels_class0 + labels_class1

    # Shuffle the combined dataset
    combined = list(zip(sentences, labels))
    random.shuffle(combined)  # Shuffle the combined list
    sentences, labels = zip(*combined)

    # Split dataset into training and test sets
    train_ratio = 0.8  # 80% training, 20% testing
    split_idx = int(len(sentences) * train_ratio)
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]

    # Create datasets and dataloaders
    context_size = 300  # Fixed context size
    train_dataset = SentenceDataset(train_sentences, train_labels, vocab, context_size)
    test_dataset = SentenceDataset(test_sentences, test_labels, vocab, context_size)
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn
    )

    # Initialize the Elman RNN model
    model = ElmanRNN(vocab_size=vocab.size, embed_size=128, hidden_dim=256)

    # Train the model
    train_model(model, train_loader, num_epochs=10, learning_rate=0.0001, device=device)

    # Evaluate the model on the test set
    test_accuracy = evaluate_model(model, test_loader, device=device)
    print(f"Test Accuracy (ElmanRNN): {test_accuracy:.4f}")
