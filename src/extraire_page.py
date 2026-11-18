from pathlib import Path

import pymupdf

racine_projet = Path(__file__).resolve().parents[1]
chemin_pdf = racine_projet / "data" / "attention.pdf"
dossier_images = racine_projet / "data" / "images"

if not chemin_pdf.is_file():
    raise FileNotFoundError(f"PDF introuvable : {chemin_pdf}")

dossier_images.mkdir(parents=True, exist_ok=True)

numero_page = 3
chemin_image = dossier_images / f"attention_page_{numero_page}.png"

with pymupdf.open(str(chemin_pdf)) as document:
    page = document[numero_page - 1]
    image = page.get_pixmap(dpi=150)
    image.save(str(chemin_image))

print(f"Image enregistrée : {chemin_image}")
print(f"Source : {chemin_pdf.name}, page {numero_page}")