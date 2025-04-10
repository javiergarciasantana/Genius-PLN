import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def read_and_tokenize(filename):
    """
    Lee un archivo, y crea una lista donde cada elemento es una lista que contiene todas las palabras de una canción.
    Sustituye los finales de línea de cada canción por el token "EOL".
    Por ejemplo, si el corpus es:

    Hola mundo
    Mi casa

    El perro
    El gato

    La salida debe ser:
    [['Hola', 'mundo', 'EOL', 'Mi', 'casa'], ['El', 'perro', 'EOL', 'El', 'gato']]

    Args:
        filename (str): Path to the input file.

    Returns:
        list: A list of sentences, where each sentence is a list of words.
    """
    sentences = []
    with open(filename, "r", encoding="utf-8-sig") as f:
        # Read the entire file content
        content = f.read()

        # Split the content into sentences using two consecutive newlines (\n\n)
        raw_sentences = content.split("###")

        # Process each sentence
        for raw_sentence in raw_sentences:
            # Replace remaining newlines within the sentence with "EOL"
            processed_sentence = raw_sentence.replace("\n", " EOL ").strip()

            # Split the sentence into words
            tokens = processed_sentence.split()

            # Remove the first "EOL" if it exists
            if tokens and tokens[0] == "EOL":
                tokens.pop(0)

            # Remove the last "EOL" if it exists
            if tokens and tokens[-1] == "EOL":
                tokens.pop()

            # Append the processed sentence if it's not empty
            if tokens:  # Ignore empty sentences
                sentences.append(tokens)

    return sentences


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


class BengioNN(nn.Module):
    def __init__(self, vocab_size, context_size, embed_size, hidden_dim):
        super(BengioNN, self).__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(context_size * embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)
        hidden = F.relu(self.fc1(embedded))
        logits = self.fc2(hidden)
        return logits


def prepare_bengio_dataset(sentences, vocab, context_size):
    data = []
    if "<PAD>" not in vocab.token_to_idx:
        vocab.add_token("<PAD>")
    pad_idx = vocab.token_to_idx["<PAD>"]
    for sentence in sentences:
        encoded_sentence = vocab.encode(sentence)
        padded_sentence = [pad_idx] * (context_size - 1) + encoded_sentence
        for i in range(len(padded_sentence) - context_size):
            inputs = padded_sentence[i : i + context_size]
            target = padded_sentence[i + context_size]
            data.append((inputs, target))
    return data


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, target = self.data[idx]
        return torch.tensor(inputs), torch.tensor(target)


def train_model(model, data_loader, vocab, num_epochs, learning_rate, device="cpu"):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token_to_idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}"
        )


def generate_language_with_top_k_sampling(
    model, start_text, max_length, k, vocab, device="cpu"
):
    input_tokens = vocab.encode(start_text)
    context_size = getattr(model, "context_size", None)

    if "<PAD>" not in vocab.token_to_idx:
        vocab.add_token("<PAD>")
    pad_idx = vocab.token_to_idx["<PAD>"]

    with torch.no_grad():
        for _ in range(max_length):
            model.to(device)
            model.eval()
            if len(input_tokens) < context_size:
                padded_input = [pad_idx] * (
                    context_size - len(input_tokens)
                ) + input_tokens
            else:
                padded_input = input_tokens[-context_size:]
            inputs = torch.tensor([padded_input], device=device)
            outputs = model(inputs)
            logits = outputs[0]

            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1).item()
            next_token = top_k_indices[next_token_idx].item()
            input_tokens.append(next_token)

            if next_token == vocab.token_to_idx.get(
                "<EOS>", None
            ) or next_token == vocab.token_to_idx.get("<PAD>", None):
                break

    return " ".join(vocab.decode(input_tokens))


# Example usage
if __name__ == "__main__":
    # Example dataset
    sentences = sentences = read_and_tokenize(r"../output/tradicional_lyrics.txt")
    sentences = sentences[1:30]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabulary
    vocab = Vocabulary()
    for sentence in sentences:
        for token in sentence:
            vocab.add_token(token)
    vocab.add_token("<PAD>")  # Add padding token

    # Prepare dataset
    context_size = 3
    bengio_data = prepare_bengio_dataset(sentences, vocab, context_size=context_size)
    bengio_loader = DataLoader(CustomDataset(bengio_data), batch_size=32, shuffle=True)

    # Initialize and train BengioNN
    bengio_nn = BengioNN(
        vocab_size=vocab.size, embed_size=128, hidden_dim=256, context_size=context_size
    )
    train_model(
        bengio_nn,
        bengio_loader,
        vocab,
        num_epochs=10,
        learning_rate=0.001,
        device=device,
    )

    # Generate text
    start_text = ["La"]
    generated_text = generate_language_with_top_k_sampling(
        bengio_nn, start_text, max_length=50, k=5, vocab=vocab, device=device
    )
    print("Generated Text:", generated_text)
