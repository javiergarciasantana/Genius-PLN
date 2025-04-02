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
    with open(filename, 'r', encoding='utf-8-sig') as f:
        # Read the entire file content
        content = f.read()

        # Split the content into sentences using two consecutive newlines (\n\n)
        raw_sentences = content.split('###')

        # Process each sentence
        for raw_sentence in raw_sentences:
            # Replace remaining newlines within the sentence with "EOL"
            processed_sentence = raw_sentence.replace('\n', ' EOL ').strip()

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

# ----------------------------- ElmanRNN Model Definition -----------------------------
class ElmanRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim):
        super(ElmanRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

# ----------------------------- Dataset Handling -----------------------------
def prepare_rnn_dataset(sentences, vocab):
    """
    Prepares a dataset for the Elman RNN model.
    Args:
        sentences: List of lists, where each inner list contains tokens for a sentence.
        vocab: Vocabulary object for encoding/decoding tokens.
    Returns:
        data: List of tuples (inputs, targets), where inputs and targets are lists of token indices.
    """
    data = []
    if '<PAD>' not in vocab.token_to_idx:
        vocab.add_token('<PAD>')
    pad_idx = vocab.token_to_idx['<PAD>']
    for sentence in sentences:
        encoded_sentence = vocab.encode(sentence)
        inputs = encoded_sentence[:-1]  # All tokens except the last
        targets = encoded_sentence[1:]  # All tokens except the first
        data.append((inputs, targets))
    return data

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, targets = self.data[idx]
        return torch.tensor(inputs), torch.tensor(targets)

def rnn_collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

# ----------------------------- Training Function -----------------------------
def train_model(model, data_loader, vocab, num_epochs, learning_rate, device='cpu'):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)  # ElmanRNN returns outputs and hidden state
            outputs = outputs.view(-1, vocab.size)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

# ----------------------------- Text Generation Function -----------------------------
def generate_language_with_top_k_sampling(model, start_text, max_length, k, vocab, device='cpu'):
    input_tokens = vocab.encode(start_text)

    if '<PAD>' not in vocab.token_to_idx:
        vocab.add_token('<PAD>')
    pad_idx = vocab.token_to_idx['<PAD>']

    with torch.no_grad():
        model.eval()
        hidden = None
        for _ in range(max_length):
            inputs = torch.tensor([input_tokens], device=device)
            outputs, hidden = model(inputs, hidden)
            logits = outputs[0, -1, :]  # Get logits for the last token

            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1).item()
            next_token = top_k_indices[next_token_idx].item()
            input_tokens.append(next_token)

            if next_token == vocab.token_to_idx.get('<EOS>', None) or next_token == vocab.token_to_idx.get('<PAD>', None):
                break

    return " ".join(vocab.decode(input_tokens))

# ----------------------------- Main Script -----------------------------
if __name__ == "__main__":
    # Example dataset
    sentences = read_and_tokenize(r"../corpus/tradicional_lyrics.txt")
    #sentences = sentences[1:30]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabulary
    vocab = Vocabulary()
    for sentence in sentences:
        for token in sentence:
            vocab.add_token(token)
    vocab.add_token('<PAD>')  # Add padding token

    # Prepare dataset
    rnn_data = prepare_rnn_dataset(sentences, vocab)
    rnn_loader = DataLoader(CustomDataset(rnn_data), batch_size=32, shuffle=True, collate_fn=rnn_collate_fn)

    # Initialize and train ElmanRNN
    elman_rnn = ElmanRNN(vocab_size=vocab.size, embed_size=128, hidden_dim=256)
    train_model(elman_rnn, rnn_loader, vocab, num_epochs=10, learning_rate=0.001, device=device)

    # Generate text
    start_text = ["La"]
    generated_text = generate_language_with_top_k_sampling(elman_rnn, start_text, max_length=50, k=5, vocab=vocab, device=device)
    print("Generated Text:", generated_text)