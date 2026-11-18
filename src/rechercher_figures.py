import json
from getpass import getpass
from pathlib import Path

import psycopg
from sentence_transformers import SentenceTransformer

racine_projet = Path(__file__).resolve().parents[1]
nom_modele = "sentence-transformers/clip-ViT-B-32"

description = input("Décris la figure recherchée en anglais : ").strip()

if not description:
    raise SystemExit("Saisis une description pour lancer la recherche.")

modele = SentenceTransformer(nom_modele, device="cpu")
vecteur = modele.encode(description, normalize_embeddings=True)
vecteur_sql = json.dumps(vecteur.tolist(), allow_nan=False)

mot_de_passe = getpass("Mot de passe PostgreSQL : ")

with psycopg.connect(
    host="127.0.0.1",
    port=5433,
    dbname="maintidoc",
    user="maintidoc",
    password=mot_de_passe,
    connect_timeout=5,
) as connexion:
    resultats = connexion.execute(
        """
        SELECT id, document, page, numero_figure, chemin_image,
               embedding <=> %s::vector AS distance
        FROM figures
        WHERE modele = %s
        ORDER BY distance, id
        LIMIT 3
        """,
        (vecteur_sql, nom_modele),
    ).fetchall()

if not resultats:
    raise SystemExit("Aucune figure indexée avec ce modèle CLIP.")

for identifiant, document, page, numero_figure, chemin_image, distance in resultats:
    similarite = 1 - distance

    print(f"\nFigure : {identifiant}")
    print(f"Source : {document}, page {page}, figure {numero_figure}")
    print(f"Similarité CLIP : {similarite:.3f}")
    print(f"Image : {racine_projet / chemin_image}")