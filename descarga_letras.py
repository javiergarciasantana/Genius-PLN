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

artist = genius.search_artist("Mestisay", sort="title")
print(artist.songs)

songs = []
# Descargar las letras de las canciones en texto plano
for song in artist.songs:
    print(f"Descargando letra de {song.title}...")
    songs.append(song.lyrics)

# Introducir todas en un solo archivo de texto con doble salto de línea
with open("mestisay_lyrics.txt", "w", encoding="utf-8") as f:
    for song in songs:
        f.write(song + "\n\n")
print("Letras guardadas en mestisay_lyrics.txt")
