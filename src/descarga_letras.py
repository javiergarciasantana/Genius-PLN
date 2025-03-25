from lyricsgenius import Genius

API_KEY = "uFdeYfD4xrNkJcT10dK429ahdfmrJ1yO-bmij0eIU8LHs4xaaHT64IyKCHJ9CISa"
genius = Genius(API_KEY)

genius.verbose = True  # Turn off status messages
genius.remove_section_headers = (
    True  # Remove section headers (e.g. [Chorus]) from lyrics when searching
)
genius.skip_non_songs = True  # Include hits thought to be non-songs (e.g. track lists)
genius.excluded_terms = [
    "(Remix)",
    "(Live)",
    "(Mixed)",
    "(FreeStyle)",
]  # Exclude songs with these words in their title
genius.sleep_time = 1
genius.retries = 3

# Elegir el corpus a incrementar
option = input("Selecciona el corpus a incrementar (0: tradicional, 1: nueva): ")
if option == "0":
    corpus = "tradicional"
elif option == "1":
    corpus = "nueva"
else:
    raise ValueError("Opción no válida")

# Elegir el artista a descargar
artist_name = input("Introduce el nombre del artista: ")
artist = genius.search_artist(artist_name, sort="title")
print(artist.songs)

songs = []
# Descargar las letras de las canciones en texto plano
for song in artist.songs:
    print(f"Descargando letra de {song.title}...")
    songs.append(song.lyrics)

# Introducir todas en un solo archivo de texto con doble salto de línea
# Abrir el archivo en modo append para no sobreescribir
with open("../${corpus}_lyrics.txt", "a", encoding="utf-8") as f:
    for song in songs:
        f.write(song + "\n\n")
print("Letras guardadas en ${corpus}_lyrics.txt")
