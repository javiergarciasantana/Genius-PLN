import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from functools import partial
import math

# ----------------------------- Tokenization and Vocabulary Management -----------------------------
def read_and_tokenize(filename):
    """
    Reads a file and tokenizes each sentence into lists of words, replacing newlines with 'EOL'.
    Args:
        filename (str): Path to the input file.
    Returns:
        list: A list of sentences, where each sentence is a list of tokens.
    """
    sentences = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        raw_sentences = content.split('###')
        for raw_sentence in raw_sentences:
            processed_sentence = raw_sentence.replace('\n', ' EOL ').strip()
            tokens = processed_sentence.split()
            if tokens and tokens[0] == "EOL":
                tokens.pop(0)
            if tokens and tokens[-1] == "EOL":
                tokens.pop()
            if tokens:
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
        unk_idx = self.token_to_idx.get('<UNK>', 0)
        return [self.token_to_idx.get(token, unk_idx) for token in tokens]

    def decode(self, indices):
        unk_token = '<UNK>'
        return [self.idx_to_token.get(idx, unk_token) for idx in indices]

# ----------------------------- Transformer Model Definition -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.heads = nn.ModuleList([
            AttentionHead(emb_dim, emb_dim // num_heads)
            for _ in range(num_heads)
        ])
        self.W_O = nn.Parameter(torch.empty(emb_dim, emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_O)

    def forward(self, x, causal_mask, padding_mask=None):
        head_outputs = [head(x, causal_mask, padding_mask) for head in self.heads]
        x_concat = torch.cat(head_outputs, dim=-1)
        return x_concat @ self.W_O

class AttentionHead(nn.Module):
    def __init__(self, emb_dim, d_h):
        super().__init__()
        self.W_Q = nn.Parameter(torch.empty(emb_dim, d_h))
        self.b_Q = nn.Parameter(torch.empty(d_h))
        self.W_K = nn.Parameter(torch.empty(emb_dim, d_h))
        self.b_K = nn.Parameter(torch.empty(d_h))
        self.W_V = nn.Parameter(torch.empty(emb_dim, d_h))
        self.b_V = nn.Parameter(torch.empty(d_h))
        self.d_h = d_h
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.zeros_(self.b_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.zeros_(self.b_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.zeros_(self.b_V)

    def forward(self, x, causal_mask, padding_mask=None):
        Q = x @ self.W_Q + self.b_Q
        K = x @ self.W_K + self.b_K
        V = x @ self.W_V + self.b_V
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)
        masked_scores = scores.masked_fill(causal_mask.unsqueeze(0) == 0, float("-inf"))
        if padding_mask is not None:
            masked_scores = masked_scores.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)
        return attention_weights @ V

class MLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or emb_dim * 4
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, feedforward_hidden_dim=None, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.mlp = MLP(emb_dim, feedforward_hidden_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, padding_mask=None):
        attn_output = self.attn(self.norm1(x), causal_mask, padding_mask)
        x = x + self.dropout(attn_output)
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_output)
        return x

class DecoderLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, num_blocks, context_size, pad_idx, dropout=0.1):
        super().__init__()
        self.context_size = context_size
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(emb_dim)
        self.output_proj = nn.Linear(emb_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x, src_key_padding_mask=None):
        batch_size, seq_len = x.size()
        x_emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        pos_enc = positional_encoding(seq_len, self.embedding.embedding_dim).to(x.device)
        x_pos = x_emb + pos_enc.unsqueeze(0)
        x_pos = self.pos_dropout(x_pos)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        hidden_states = x_pos
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask, src_key_padding_mask)
        hidden_states = self.final_norm(hidden_states)
        logits = self.output_proj(hidden_states)
        return logits

def positional_encoding(max_len, d_model):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# ----------------------------- Dataset Preparation -----------------------------
def prepare_transformer_dataset(sentences, vocab, context_size):
    data = []
    if '<PAD>' not in vocab.token_to_idx:
        vocab.add_token('<PAD>')
    pad_idx = vocab.token_to_idx['<PAD>']
    for sentence in sentences:
        encoded_sentence = vocab.encode(sentence)
        for i in range(0, len(encoded_sentence), context_size):
            chunk = encoded_sentence[i:i + context_size + 1]
            if len(chunk) < 2:
                continue
            inputs = chunk[:-1]
            targets = chunk[1:]
            data.append((inputs, targets))
    return data

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, targets = self.data[idx]
        return inputs, targets

def transformer_collate_fn(batch, pad_idx):
    inputs_list, targets_list = zip(*batch)
    max_len_in_batch = max(len(seq) for seq in inputs_list)
    padded_inputs = [seq + [pad_idx] * (max_len_in_batch - len(seq)) for seq in inputs_list]
    padded_targets = [seq + [pad_idx] * (max_len_in_batch - len(seq)) for seq in targets_list]
    inputs_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    targets_tensor = torch.tensor(padded_targets, dtype=torch.long)
    src_key_padding_mask = (inputs_tensor == pad_idx)
    return inputs_tensor, targets_tensor, src_key_padding_mask

# ----------------------------- Training and Evaluation -----------------------------
def train_model(model, data_loader, vocab, num_epochs, learning_rate, device='cpu', clip_value=1.0):
    model.to(device)
    model.train()
    pad_idx = vocab.token_to_idx['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets, src_key_padding_mask in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            outputs_flat = outputs.view(-1, len(vocab.token_to_idx))
            targets_flat = targets.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        perplexity = math.exp(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

def compute_perplexity(model, data_loader, vocab, device='cpu'):
    model.to(device)
    model.eval()
    total_loss = 0
    total_tokens = 0
    pad_idx = vocab.token_to_idx['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
    with torch.no_grad():
        for inputs, targets, src_key_padding_mask in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            outputs_flat = outputs.view(-1, len(vocab.token_to_idx))
            targets_flat = targets.view(-1)
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            total_tokens += (targets_flat != pad_idx).sum().item()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# ----------------------------- Text Generation -----------------------------
def generate_language_with_top_k_sampling(model, start_text, max_length, k, vocab, device='cpu'):
    model.to(device)
    model.eval()
    input_tokens = vocab.encode(start_text)
    generated_tokens = list(input_tokens)
    pad_idx = vocab.token_to_idx['<PAD>']
    eos_idx = vocab.token_to_idx.get('<EOS>', -1)
    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor([generated_tokens[-model.context_size:]], dtype=torch.long, device=device)
            outputs = model(inputs)
            next_token_logits = outputs[0, -1, :]
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = top_k_indices[torch.multinomial(probs, 1).item()].item()
            generated_tokens.append(next_token_idx)
            if next_token_idx == eos_idx or next_token_idx == pad_idx:
                break
    return " ".join(vocab.decode(generated_tokens))


def save_model(model, filepath):
  """
  Saves the model's state dictionary to the specified file.

  Args:
      model (nn.Module): The PyTorch model to save.
      filepath (str): Path where the model will be saved.
  """
  torch.save(model.state_dict(), filepath)
  print(f"Model saved successfully to {filepath}")



# ----------------------------- Main Script -----------------------------
if __name__ == "__main__":
    import os
    os.makedirs("saved_models", exist_ok=True)

    FILENAME = "../nueva_lyrics.txt"
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Leer datos y construir vocabulario
    sentences = read_and_tokenize(FILENAME)
    vocab = Vocabulary()
    for sentence in sentences:
        for token in sentence:
            vocab.add_token(token)
    vocab.add_token('<PAD>')

    # Definir hiperparámetros a probar
    CONTEXT_SIZES = [32, 64]
    BATCH_SIZES = [32, 64]
    NUM_EPOCHS_LIST = [10]
    LEARNING_RATES = [0.001, 0.0005]

    for CONTEXT_SIZE in CONTEXT_SIZES:
        for BATCH_SIZE in BATCH_SIZES:
            for NUM_EPOCHS in NUM_EPOCHS_LIST:
                for LEARNING_RATE in LEARNING_RATES:
                    print(f"\n🔧 Testing: CONTEXT={CONTEXT_SIZE}, BATCH={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}, LR={LEARNING_RATE}")

                    # Preparar dataset y dataloader
                    dataset = prepare_transformer_dataset(sentences, vocab, CONTEXT_SIZE)
                    collate_fn = partial(transformer_collate_fn, pad_idx=vocab.token_to_idx['<PAD>'])
                    data_loader = DataLoader(CustomDataset(dataset), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

                    # Inicializar modelo
                    model = DecoderLanguageModel(
                        vocab_size=len(vocab.token_to_idx),
                        emb_dim=128,
                        num_heads=4,
                        num_blocks=4,
                        context_size=CONTEXT_SIZE,
                        pad_idx=vocab.token_to_idx['<PAD>']
                    ).to(DEVICE)

                    # Entrenamiento
                    train_model(model, data_loader, vocab, NUM_EPOCHS, LEARNING_RATE, DEVICE)

                    # Evaluación
                    perplexity = compute_perplexity(model, data_loader, vocab, DEVICE)
                    print(f"📉 Perplexity: {perplexity:.4f}")

                    # Guardar modelo
                    model_filename = f"model_ctx{CONTEXT_SIZE}_bs{BATCH_SIZE}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}.pth"
                    save_model(model, os.path.join("saved_models", model_filename))

                    # Generación de texto
                    prompt = ["Que", "es", "el", "amor", "?"]
                    generated_text = generate_language_with_top_k_sampling(model, prompt, 50, 5, vocab, DEVICE)
                    print("📝 Generated Text:", generated_text)
