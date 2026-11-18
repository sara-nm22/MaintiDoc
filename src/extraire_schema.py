from pathlib import Path

import pymupdf

racine_projet = Path(__file__).resolve().parents[1]
chemin_pdf = racine_projet / "data" / "attention.pdf"
dossier_images = racine_projet / "data" / "images"

if not chemin_pdf.is_file():
    raise FileNotFoundError(f"PDF introuvable : {chemin_pdf}")

dossier_images.mkdir(parents=True, exist_ok=True)

numero_page = 3
chemin_image = dossier_images / "attention_page_3_figure_1.png"

with pymupdf.open(str(chemin_pdf)) as document:
    page = document[numero_page - 1]
    zone_schema = pymupdf.Rect(185, 60, 425, 425)
    image = page.get_pixmap(clip=zone_schema, dpi=200)
    image.save(str(chemin_image))

print(f"Schéma enregistré : {chemin_image}")
print(f"Source : {chemin_pdf.name}, page {numero_page}, figure 1")