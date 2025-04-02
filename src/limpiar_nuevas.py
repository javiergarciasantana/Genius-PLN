import re

# Nombre del archivo original y del archivo limpio
archivo_original = "../corpus/nueva_lyrics.txt"  # Ajusta según corresponda
archivo_limpio = "../corpus/nueva_lyrics_limpio.txt"

# Patrón para detectar líneas no deseadas
patrones_a_eliminar = [
    r"^\s*\d*\s*Contributors.*?\n",  # "9 ContributorsDonde Están Lyrics"
    r"^\s*Translations.*?\n",        # "TranslationsEnglish"
    r"^\s*\d*\s*Lyrics.*?\n",        # "1 ContributorDonde Están Lyrics"
    r"^\s*\d*\s*[A-Za-z]+ FEBRERO.*?\n",  # "14 FEBRERO"
]

# Leer el archivo original y limpiar líneas no deseadas
with open(archivo_original, "r", encoding="utf-8") as f:
    lineas = f.readlines()

# Filtrar las líneas usando regex
lineas_limpias = []
for linea in lineas:
    if not any(re.match(patron, linea) for patron in patrones_a_eliminar):
        lineas_limpias.append(linea)

# Guardar el resultado en un nuevo archivo
with open(archivo_limpio, "w", encoding="utf-8") as f:
    f.writelines(lineas_limpias)

print(f"Archivo limpio guardado como {archivo_limpio}")
