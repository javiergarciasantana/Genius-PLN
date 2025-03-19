import csv
from lyricsgenius import Genius

# Cambiar api
genius = Genius("_qXFZ4xqWmA8Qgi6XDN73aJUGirIZmPX7yyt7AhJW1VxlagILLXDJfb4li5HtV8m")

genius.verbose = True # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
genius.skip_non_songs = True # Include hits thought to be non-songs (e.g. track lists)
genius.excluded_terms = ["(Remix)", "(Live)", "(Mixed)", "(FreeStyle)"] # Exclude songs with these words in their title
genius.sleep_time=1
genius.retries=3

artist = genius.search_artist("Mestisay", sort="title")

csv_filename = "Mestisay_lyrics.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Lyrics"])  # Header row

    for song in artist.songs:
        writer.writerow([song.title, song.lyrics])  # Write song title and lyrics

print(f"Data saved to {csv_filename}")