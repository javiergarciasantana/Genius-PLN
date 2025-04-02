import re

# Nombre del archivo original y del archivo limpio
archivo_original = "../corpus/nueva_lyrics.txt"
archivo_limpio = "../corpus/nueva_lyrics_limpio.txt"

# Expresión regular mejorada para detectar líneas con "Contributors"
patron_contributors = re.compile(r"^\s*\d*\s*Contributor.*$", re.MULTILINE)

# Leer el archivo original
with open(archivo_original, "r", encoding="utf-8") as f:
    lineas = f.readlines()

# Filtrar las líneas eliminando completamente las que coincidan
lineas_limpias = [
    linea for linea in lineas if not patron_contributors.match(linea.strip())
]

# Guardar el resultado sin líneas vacías
with open(archivo_limpio, "w", encoding="utf-8") as f:
    f.writelines(lineas_limpias)

print(f"✅ Archivo limpio guardado como {archivo_limpio}")
