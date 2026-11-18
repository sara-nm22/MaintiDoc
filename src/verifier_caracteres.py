import json
from pathlib import Path

racine_projet = Path(__file__).resolve().parents[1]
chemin_json = racine_projet / "data" / "processed" / "attention_pages.json"

with chemin_json.open("r", encoding="utf-8") as fichier:
    pages = json.load(fichier)

caracteres = [chr(57346), chr(57421)]

for page in pages:
    texte = page["texte"]

    for position, caractere in enumerate(texte):
        if caractere in caracteres:
            debut = max(0, position - 50)
            fin = position + 51
            contexte = texte[debut:fin].replace("\n", " ")

            print(f"\nPage {page['page']} — U+{ord(caractere):04X}")
            print(contexte)