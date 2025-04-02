import torch
import torch.nn.functional as F
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize

import numpy as np


def read_and_tokenize_wordvec(filename, preprocess=None):
    """
    Lee un archivo, y crea una lista donde cada elemento es una lista que contiene todas las palabras de una canción.
    Aplica una función de preprocesamiento opcional al contenido completo del archivo.

    Args:
        filename (str): Path to the input file.
        preprocess (function, optional): A function to preprocess the content. Defaults to None.

    Returns:
        list: A list of sentences, where each sentence is a list of words.
    """
    with open(filename, "r", encoding="utf-8-sig") as f:
        # Read the entire file content
        content = f.read()

        # Apply preprocessing function if provided
        if preprocess is not None:
            content = preprocess(content)

        # Split the content into sentences using two consecutive newlines (\n\n)
        raw_sentences = content.split("\n")

        # Process each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # Split the sentence into words
            tokens = raw_sentence.split()

            # Append the processed sentence if it's not empty
            if tokens:  # Ignore empty sentences
                sentences.append(tokens)

    return sentences


# Example Preprocessing Function
def preprocess(content):
    """
    Preprocessing function to:
    - Replace newlines within sentences with "EOL"
    - Replace "\n\n" with "\n"
    - Remove punctuation including ? ¿ ! ¡
    - Convert to lowercase
    """
    # Replace remaining newlines within the content with "EOL"
    content = content.replace("\n\n", "\n")

    # Define additional punctuation to remove
    additional_punctuation = "?¿!¡"

    # Combine standard punctuation with additional punctuation
    all_punctuation = string.punctuation + additional_punctuation

    # Remove punctuation (except for "EOL")
    content = content.translate(str.maketrans("", "", all_punctuation))

    # Convert to lowercase
    content = content.lower()

    # Strip leading/trailing whitespace
    content = content.strip()

    return content


def write_most_similar_words_gensim(model, output_file):
    """
    Escribe las palabras más similares para cada palabra en el vocabulario en un archivo.
    Incluye el valor de similitud entre paréntesis después de cada palabra.

    Args:
        model (Word2Vec): Modelo Word2Vec entrenado.
        output_file (str): Ruta del archivo de salida.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        for word in model.wv.index_to_key:
            try:
                # Obtener las 10 palabras más similares y sus valores de similitud
                most_similar = model.wv.most_similar(word, topn=10)

                # Formatear las palabras similares como "palabra (similitud)"
                similar_words_str = ", ".join(
                    [f"{w} ({s:.2f})" for w, s in most_similar]
                )

                # Escribir la palabra original y sus palabras similares en el archivo
                file.write(f"{word} -> {similar_words_str}\n")

            except KeyError:
                # Si no se encuentran palabras similares, escribir un mensaje indicando esto
                file.write(f"{word} -> No similar words found\n")


def generate_most_similar_words_file(embeddings, vocab, output_file, top_k=10):
    embeddings = (
        torch.tensor(embeddings)
        if not isinstance(embeddings, torch.Tensor)
        else embeddings
    )
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

    with open(output_file, "w", encoding="utf-8") as f:
        for word in vocab.reverse_vocab.values():
            word_idx = vocab.get_index(word)
            word_embedding = normalized_embeddings[word_idx]
            similarities = torch.matmul(normalized_embeddings, word_embedding)
            similarities[word_idx] = -1
            top_k_values, top_k_indices = torch.topk(similarities, top_k)
            top_k_values = top_k_values.tolist()
            top_k_indices = top_k_indices.tolist()
            similar_words = [vocab.reverse_vocab[idx] for idx in top_k_indices]
            similar_words_with_scores = [
                f"{w} ({sim:.4f})" for w, sim in zip(similar_words, top_k_values)
            ]
            f.write(f"{word}: {', '.join(similar_words_with_scores)}\n")
    print(f"Las palabras más similares han sido guardadas en '{output_file}'.")


# Paso 7: Visualización de embeddings
def visualize_embeddings_gensim(model, min_frequency=10, method="pca"):
    """
    Visualiza embeddings usando PCA o t-SNE, filtrando palabras con poca frecuencia y stopwords.

    Args:
        model (Word2Vec): Modelo Word2Vec entrenado.
        min_frequency (int): Frecuencia mínima para incluir una palabra en la visualización.
        method (str): Método de visualización ("pca" o "tsne").
    """
    # Obtener stopwords en español
    stop_words = set(stopwords.words("spanish"))

    # Filtrar palabras: solo incluir palabras con frecuencia mayor a min_frequency y no stopwords
    words = [
        word
        for word in model.wv.index_to_key
        if model.wv.get_vecattr(word, "count") >= min_frequency
        and word.lower() not in stop_words
    ]

    # Limitar el número de palabras para facilitar la visualización (opcional)
    words = words[:100]  # Seleccionar las primeras 100 palabras más frecuentes

    # Extraer los embeddings correspondientes a las palabras filtradas
    word_embeddings = np.array([model.wv[word] for word in words])

    # Reducción de dimensionalidad
    if method == "pca":
        reducer = PCA(n_components=2)
        title = "Visualización de Embeddings con PCA"
        xlabel, ylabel = "Componente Principal 1", "Componente Principal 2"
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, metric="cosine")
        title = "Visualización de Embeddings con t-SNE"
        xlabel, ylabel = "Dimensión 1", "Dimensión 2"
    else:
        raise ValueError("Método no válido. Usa 'pca' o 'tsne'.")

    reduced_embeddings = reducer.fit_transform(normalize(word_embeddings))

    # Visualizar los embeddings
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y)
        plt.text(x, y, word, fontsize=9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def visualize_embeddings(
    word_embeddings, words, method="pca", perplexity=30, random_state=42
):
    """
    Visualizes word embeddings after applying a specified dimensionality reduction method.

    Args:
        word_embeddings (numpy.ndarray): The word embeddings to visualize (num_words x embedding_dim).
        words (list): List of words corresponding to the embeddings.
        method (str): Dimensionality reduction method ("pca" or "tsne").
        perplexity (float): Perplexity parameter for t-SNE (only used if method="tsne").
        random_state (int): Random state for reproducibility (only used if method="tsne").

    Returns:
        None
    """
    # Apply PCA transformation
    if method == "pca":
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(normalize(word_embeddings))
        title = "PCA Visualization"
        xlabel = "Principal Component 1"
        ylabel = "Principal Component 2"

    # Apply t-SNE transformation
    elif method == "tsne":
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            metric="cosine",
        )
        reduced_embeddings = tsne.fit_transform(normalize(word_embeddings))
        title = "t-SNE Visualization"
        xlabel = "t-SNE Dimension 1"
        ylabel = "t-SNE Dimension 2"

    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'pca' or 'tsne'.")

    # Plot the reduced embeddings
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y)
        plt.text(x, y, word, fontsize=9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


from gensim.models.callbacks import CallbackAny2Vec


class LossLogger(CallbackAny2Vec):
    """
    Custom callback to log loss after each epoch.
    """

    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, model):
        """
        Called at the end of each epoch.
        """
        # Get the latest loss
        loss = model.get_latest_training_loss()

        # Log the loss
        print(f"Epoch {self.epoch}, Loss: {loss}")
        self.losses.append(loss)

        # Reset the loss for the next epoch
        model.running_training_loss = 0.0

        self.epoch += 1


loss_logger = LossLogger()


from gensim.models import KeyedVectors
import io

# Load gensim word2vec


def write_tsv(w2v_path):
    # Cargar el modelo Word2Vec
    w2v = KeyedVectors.load_word2vec_format(w2v_path)

    # Obtener la lista de stopwords (en español, por ejemplo)
    stop_words = set(stopwords.words("spanish"))

    # Archivos de salida
    out_v = io.open("vecs.tsv", "w", encoding="utf-8")  # Archivo de vectores
    out_m = io.open("meta.tsv", "w", encoding="utf-8")  # Archivo de metadatos

    # Iterar sobre todas las palabras del modelo
    for index in range(len(w2v.index_to_key)):
        word = w2v.index_to_key[index]
        vec = w2v.vectors[index]

        # Filtrar stopwords
        if word not in stop_words:
            # Escribir la palabra en el archivo de metadatos
            out_m.write(word + "\n")
            # Escribir el vector en el archivo de vectores
            out_v.write("\t".join([str(x) for x in vec]) + "\n")

    # Cerrar los archivos
    out_v.close()
    out_m.close()
