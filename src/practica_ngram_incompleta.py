import random
import math
import os
from collections import defaultdict


def read_and_tokenize(filename) -> list:
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


def write_sentences_to_file(sentences, output_file):
    """
    Escribe una lista de listas en un archivo de texto, donde cada lista se escribe en una línea.

    Args:
        sentences (list of list of str): Lista de listas de palabras.
        output_file (str): Ruta del archivo de salida.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for word_list in sentences:
            # Convierte la lista de palabras en una cadena separada por espacios
            line = " ".join(word_list)
            # Escribe la línea en el archivo y agrega un salto de línea
            f.write(line + "\n")


def prepare_corpus(corpus, n, unk_threshold=-1):
    """
    Prepara el corpus agregando tokens <s> al inicio y </s> al final de cada oración.
    Reemplaza palabras poco frecuentes con <UNK> y construye el vocabulario.

    Args:
        corpus (list of list of str): El corpus tokenizado.
        n (int): El tamaño del modelo de n-gramas.
        unk_threshold (int): Palabras con frecuencia <= este umbral se reemplazan con <UNK>.

    Returns:
        tuple: Una tupla que contiene el corpus procesado (lista de listas de str) y el vocabulario (set).
    """
    word_counts = {}
    for sentence in corpus:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Construir el vocabulario excluyendo palabras poco frecuentes
    vocab = {word for word, count in word_counts.items() if count > unk_threshold}
    # Agregar tokens especiales al vocabulario
    if unk_threshold >= 0:
        vocab.update({"<s>", "</s>", "<UNK>"})
    else:
        vocab.update({"<s>", "</s>"})

    processed_corpus = []
    for sentence in corpus:
        # Agregar (n-1) tokens <s> al inicio y </s> al final
        processed_sentence = (
            ["<s>"] * (n - 1)
            + [word if word in vocab else "<UNK>" for word in sentence]
            + ["</s>"]
        )
        processed_corpus.append(processed_sentence)
    return processed_corpus, vocab


def generate_ngrams(corpus, n):
    """
    Genera todos los n-gramas contiguos del corpus preparado.

    Args:
        corpus (list of list of str): El corpus preparado.
        n (int): El tamaño de los n-gramas.

    Returns:
        list: Una lista de n-gramas, donde cada n-grama es una tupla de longitud n.
    """
    ngrams = []
    for sentence in corpus:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])
            ngrams.append(ngram)
    return ngrams


def compute_ngram_probabilities(ngrams, vocab, n):
    """
    Calcula las probabilidades suavizadas de Laplace para todos los n-gramas observados.
    Muestra mensajes de progreso durante el cálculo.

    Args:
        ngrams (list of tuple): Lista de n-gramas generados del corpus.
        vocab (set): Conjunto de palabras en el vocabulario.
        n (int): El tamaño de los n-gramas.

    Returns:
        dict: Un diccionario donde las claves son contextos (tuplas de longitud n-1)
              y los valores son distribuciones de probabilidad suavizadas.
    """
    cfd = defaultdict(
        lambda: defaultdict(int)
    )  # Distribución condicional de frecuencias
    print(f"Procesando {len(ngrams)} n-gramas...")
    for i, ngram in enumerate(ngrams):
        context, target = tuple(ngram[:-1]), ngram[-1]
        if target == "<s>":  # Ignorar <s> como objetivo
            continue
        cfd[context][target] += 1
        # Mostrar progreso cada 10,000 n-gramas
        if (i + 1) % 10000 == 0:
            print(f"Procesados {i + 1}/{len(ngrams)} n-gramas...")

    # Crear una lista de palabras objetivo excluyendo <s>
    target_vocab = list(vocab - {"<s>"})
    vocab_size = len(target_vocab)
    print(f"Calculando probabilidades para {len(cfd)} contextos observados...")
    cpd = {}
    for j, (context, observed_counts) in enumerate(cfd.items()):
        total_count = sum(observed_counts.values())
        # Calcular probabilidades suavizadas de Laplace
        smoothed_probs = {
            word: (observed_counts.get(word, 0) + 1) / (total_count + vocab_size)
            for word in target_vocab
        }
        cpd[context] = smoothed_probs
        # Mostrar progreso cada 100 contextos
        if (j + 1) % 100 == 0:
            print(f"Procesados {j + 1}/{len(cfd)} contextos...")
    print("Cálculo de probabilidades completado.")
    return cpd


def sentence_logprobability(sentence, cpd, n, vocab):
    """
    Calcula la probabilidad logarítmica de una oración usando el modelo de n-gramas con suavizado de Laplace.
    Reemplaza palabras desconocidas con <UNK> y maneja contextos no observados de manera robusta.

    Args:
        sentence (list of str): La oración como una lista de palabras.
        cpd (dict): Diccionario de distribuciones de probabilidad condicional.
        n (int): El tamaño de los n-gramas.
        vocab (set): Conjunto de palabras en el vocabulario.

    Returns:
        float: La probabilidad logarítmica de la oración.
    """
    # Reemplazar palabras fuera del vocabulario con <UNK>
    processed_sentence = [word if word in vocab else "<UNK>" for word in sentence]
    log_prob = 0.0
    for i in range(len(processed_sentence) - n + 1):
        context = tuple(processed_sentence[i : i + n - 1])
        target = processed_sentence[i + n - 1]
        if context in cpd:
            # Usar <UNK> si el objetivo no fue observado en este contexto
            token = target if target in cpd[context] else "<UNK>"
            prob = cpd[context].get(
                token, 1 / len(vocab)
            )  # Probabilidad uniforme para <UNK>
        else:
            # Aplicar suavizado de Laplace para contextos no observados
            prob = 1 / len(list(vocab - {"<s>"}))
        log_prob += math.log(prob)
    return log_prob


def generate_sentence_top_k(
    cpd, vocab, n, max_length=20, special_tokens_to_remove={"<s>", "</s>"}, k=20
):
    """
    Genera una oración usando muestreo top-k.
    Comienza con (n-1) tokens <s> y muestrea la siguiente palabra hasta generar </s> o alcanzar la longitud máxima.

    Args:
        cpd (dict): Diccionario de distribuciones de probabilidad condicional.
        vocab (set): Conjunto de palabras en el vocabulario.
        n (int): El tamaño de los n-gramas.
        max_length (int): Longitud máxima de la oración generada.
        special_tokens_to_remove (set): Tokens a eliminar del resultado final.
        k (int): Número de palabras más probables a considerar durante el muestreo.

    Returns:
        list: La oración generada sin tokens especiales.
    """
    # Inicializar la oración con (n-1) tokens <s>
    context = ("<s>",) * (n - 1)
    sentence = list(context)
    while len(sentence) < max_length:
        if context in cpd:
            # Obtener la distribución de probabilidad para el contexto actual
            probabilities = cpd[context]
            # Seleccionar las k palabras más probables
            top_k_words = sorted(
                probabilities.items(), key=lambda x: x[1], reverse=True
            )[:k]
            words, probs = zip(*top_k_words)
            # Normalizar probabilidades para las k palabras seleccionadas
            total_prob = sum(probs)
            normalized_probs = [p / total_prob for p in probs]
            # Muestrear la siguiente palabra basada en probabilidades normalizadas
            next_word = random.choices(words, weights=normalized_probs, k=1)[0]
        else:
            # Si el contexto no está en cpd, muestrear una palabra al azar del vocabulario
            next_word = random.choice(list(vocab))
        # Agregar la palabra muestreada a la oración
        sentence.append(next_word)
        # Actualizar el contexto
        context = tuple(sentence[-(n - 1) :])
        # Detener la generación si se alcanza </s>
        if next_word == "</s>":
            break
        # Remover tokens especiales del resultado final
        tokens_return = [
            token for token in sentence if token not in special_tokens_to_remove
        ]
        processed_words = []

        for word in tokens_return:
            if word == "EOL":
                # Sustituir 'EOL' por un salto de línea
                processed_words.append("\n")
            else:
                # Mantener la palabra tal como está
                processed_words.append(word)

    # Unir las palabras con espacios y retornar el texto generado
    return " ".join(processed_words)


def split_corpus(corpus, train_ratio=0.8):
    """
    Divide el corpus en conjuntos de entrenamiento y prueba.

    Args:
        corpus (list of list of str): El corpus preparado.
        train_ratio (float): Proporción del corpus para entrenamiento (default: 0.8).

    Returns:
        tuple: Una tupla que contiene el corpus de entrenamiento y el corpus de prueba.
    """
    split_index = int(len(corpus) * train_ratio)
    train_corpus = corpus[:split_index]
    test_corpus = corpus[split_index:]

    return train_corpus, test_corpus


def compute_perplexity(test_sentences, cpd, n, vocab):
    """
    Calcula la perplejidad del modelo sobre un conjunto de oraciones de prueba.

    Args:
        test_sentences (list of list of str): Lista de oraciones de prueba.
        cpd (dict): Diccionario de distribuciones de probabilidad condicional.
        n (int): El tamaño de los n-gramas.
        vocab (set): Conjunto de palabras en el vocabulario.

    Returns:
        float: La perplejidad del modelo.
    """
    total_log_prob = 0.0
    for sentence in test_sentences:
        total_log_prob += sentence_logprobability(sentence, cpd, n, vocab)

    # Calcular la perplejidad como la probabilidad media inversa de las oraciones
    perplexity = math.exp(-total_log_prob / sum(len(s) for s in test_sentences))
    return perplexity


# Ejecución Principal
if __name__ == "__main__":

    # Paso 1: Leer y tokenizar el corpus
    corpus = read_and_tokenize("../corpus/nueva_lyrics.txt")

    do_KenLM = True
    if do_KenLM:
        write_sentences_to_file(corpus, "output_nueva.txt")

    # Paso 2: Preparar el corpus y el vocabulario
    n = 2 # Orden del modelo de n-gramas
    unk_threshold = 0
    print("Preparando el corpus...")
    prepared_corpus, vocab = prepare_corpus(corpus, n, unk_threshold)

    # Paso 3: Dividir el corpus en entrenamiento y prueba
    do_split_corpus = True
    if do_split_corpus:
        print("Dividiendo el corpus en entrenamiento y prueba...")
        train_corpus, test_corpus = split_corpus(prepared_corpus, train_ratio=0.6)
    else:
        train_corpus = test_corpus = prepared_corpus

    # Paso 4: Generar n-gramas del corpus de entrenamiento
    print("Generando n-gramas del corpus de entrenamiento...")
    train_ngrams = generate_ngrams(train_corpus, n)

    # Paso 5: Calcular probabilidades de n-gramas
    print("Calculando probabilidades de n-gramas...")
    cpd = compute_ngram_probabilities(train_ngrams, vocab, n)

    # Paso 6: Evaluar oraciones de prueba
    print("\nEvaluando oraciones de prueba...")
    for test_sentence in test_corpus[
        :2
    ]:  # Mostrar resultados para las primeras 5 oraciones
        prob = sentence_logprobability(test_sentence, cpd, n, vocab)
        print("\nLog Probabilidad de la Oración:")
        print(" ".join(test_sentence), f"= {prob:.8f}")
        print("\nProbabilidad de la Oración:")
        print(" ".join(test_sentence), f"= {math.exp(prob):.8f}")
        print()

    # Paso 7: Generando oraciones de prueba

    for idx in range(3):
        print(f"Ejemplo {idx}")
        print(
            generate_sentence_top_k(
                cpd,
                vocab,
                n,
                max_length=300,
                special_tokens_to_remove={"<s>", "</s>"},
                k=5,
            )
        )
        print()

    # Paso 8: Calcular la perplejidad
    print("\nCalculando la perplejidad del modelo...")
    perplexity = compute_perplexity(test_corpus, cpd, n, vocab)
    print(f"Perplejidad del modelo: {perplexity:.4f}")
