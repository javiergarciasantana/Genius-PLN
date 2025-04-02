from lyricsgenius import Genius

API_KEY = "ZaKTY_An-CIrpig3OjDEYKYGE8f1MLnkf0rljTWjdU402o1YMN8rHIOnPUEELQrk"
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
    "(Original Version)",
]  # Exclude songs with these words in their title
genius.sleep_time = 7  # Wait 7 seconds between requests
genius.retries = 3

# Elegir el corpus a incrementar
option = input("Selecciona el corpus a incrementar (0: tradicional, 1: nueva): ")
corpus = ""
if option == "0":
    corpus = "tradicional"
elif option == "1":
    corpus = "nueva"
else:
    raise ValueError("Opción no válida")

# Elegir el artista a descargar
new_artist_names = ["Quevedo", "La Pantera", "Ptazeta", "Adexe y Nau", "Maikel Delacalle",
                    "Bejo", "Sara Socas", "Don Patricio", "Cruz Cafuné", "El Ima", "Juseph",
                    "Danny Romero", "Agoney"]

old_artist_names = ["Mestisay", "Los Gofiones", "Los Sabandeños", "Taburiente",
                    "Los Faycanes", "Añoranza", "Benito Cabrera", "Guadarfía",
                    "Los Huaracheros", "José Veléz"]

def downloadSong(option) -> None:
    """
      Descarga las letras de las canciones de los artistas seleccionados y las guarda en un archivo de texto.

      Args:
        option (str): Opción seleccionada por el usuario.

      Returns:
        None
    """
    songs = []
    artists_names = old_artist_names if option == "0" else new_artist_names
    for artist in artists_names:
        artist = genius.search_artist(artist, sort="title")
        if artist is None:
          print(f"Error: No se encontraron resultados para {artist}")
          continue  # Skip this iteration if no artist was found

        print(artist.songs)
        # Descargar las letras de las canciones en texto plano
        for song in artist.songs:
            print(f"Descargando letra de {song.title}...")
            songs.append(song.lyrics)
        with open(f"../{corpus}_lyrics.txt", "a", encoding="utf-8") as f:
            for song in songs:
                f.write(song + "\n" + "#" * 3 + "\n")

        print(f"Letras guardadas en {corpus}_lyrics.txt")
    
downloadSong(option)
