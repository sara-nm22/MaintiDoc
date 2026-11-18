import json
from pathlib import Path

import pymupdf

racine_projet = Path(__file__).resolve().parents[1]
chemin_pdf = racine_projet / "data" / "attention.pdf"
dossier_sortie = racine_projet / "data" / "processed"
chemin_json = dossier_sortie / "attention_tableau_2.json"

numero_page = 8
bandes_lignes = [(218, 229.5), (229.5, 241)]
colonnes = [
    ("modele", 130, 290),
    ("bleu_en_de", 290, 320),
    ("bleu_en_fr", 330, 360),
]

lignes = []

with pymupdf.open(str(chemin_pdf)) as pdf:
    page = pdf[numero_page - 1]

    for haut, bas in bandes_lignes:
        ligne = {}

        for nom_colonne, gauche, droite in colonnes:
            zone = pymupdf.Rect(gauche, haut, droite, bas)
            cellule = page.get_text("text", clip=zone).strip()

            if not cellule:
                raise ValueError(f"Cellule vide pour {nom_colonne}, page {numero_page}")

            ligne[nom_colonne] = cellule

        ligne["bleu_en_de"] = float(ligne["bleu_en_de"])
        ligne["bleu_en_fr"] = float(ligne["bleu_en_fr"])
        lignes.append(ligne)

tableau = {
    "document": chemin_pdf.name,
    "page": numero_page,
    "type": "tableau",
    "numero_tableau": 2,
    "lignes": lignes,
}

dossier_sortie.mkdir(parents=True, exist_ok=True)

with chemin_json.open("w", encoding="utf-8") as fichier:
    json.dump(tableau, fichier, ensure_ascii=False, indent=2)

print(f"Source : {chemin_pdf.name}, page {numero_page}, tableau 2")
for ligne in lignes:
    print(f"{ligne['modele']} | EN-DE {ligne['bleu_en_de']} | EN-FR {ligne['bleu_en_fr']}")
print(f"Fichier créé : {chemin_json}")