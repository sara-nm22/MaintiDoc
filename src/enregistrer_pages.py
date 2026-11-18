import json
from pathlib import Path

from llama_index.readers.file import PyMuPDFReader

racine_projet = Path(__file__).resolve().parents[1]
chemin_pdf = racine_projet / "data" / "attention.pdf"
dossier_sortie = racine_projet / "data" / "processed"
chemin_json = dossier_sortie / "attention_pages.json"

if not chemin_pdf.is_file():
    raise FileNotFoundError(f"PDF introuvable : {chemin_pdf}")

lecteur = PyMuPDFReader()
pages = lecteur.load_data(file_path=str(chemin_pdf), metadata=True)

pages_exportees = []

for page in pages:
    donnees_page = {
        "document": chemin_pdf.name,
        "page": int(page.metadata["source"]),
        "type": "texte",
        "texte": page.text,
    }
    pages_exportees.append(donnees_page)

dossier_sortie.mkdir(parents=True, exist_ok=True)

with chemin_json.open("w", encoding="utf-8") as fichier:
    json.dump(pages_exportees, fichier, ensure_ascii=False, indent=2)

print(f"Pages enregistrées : {len(pages_exportees)}")
print(f"Fichier créé : {chemin_json}")