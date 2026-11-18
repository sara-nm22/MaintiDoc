import json
from getpass import getpass
from pathlib import Path

import psycopg
from PIL import Image
from sentence_transformers import SentenceTransformer

racine = Path(__file__).resolve().parents[1]
chemin_relatif = Path("data/images/attention_page_3_figure_1.png")
chemin_image = racine / chemin_relatif

if not chemin_image.is_file():
    raise FileNotFoundError(f"Schéma introuvable : {chemin_image}")

nom_modele = "sentence-transformers/clip-ViT-B-32"
modele = SentenceTransformer(nom_modele, device="cpu")

with Image.open(chemin_image) as fichier:
    image = fichier.convert("RGB")

vecteur = modele.encode(image, normalize_embeddings=True)
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
    connexion.execute("""
        CREATE TABLE IF NOT EXISTS figures (
            id TEXT PRIMARY KEY,
            document TEXT NOT NULL,
            page INTEGER NOT NULL,
            numero_figure INTEGER NOT NULL,
            chemin_image TEXT NOT NULL,
            modele TEXT NOT NULL,
            embedding vector(512) NOT NULL
        )
    """)

    resultat = connexion.execute(
        """
        INSERT INTO figures (
            id, document, page, numero_figure, chemin_image, modele, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (id) DO UPDATE SET
            chemin_image = EXCLUDED.chemin_image,
            modele = EXCLUDED.modele,
            embedding = EXCLUDED.embedding
        RETURNING id, document, page, numero_figure, vector_dims(embedding)
        """,
        (
            "attention_figure_01",
            "attention.pdf",
            3,
            1,
            chemin_relatif.as_posix(),
            nom_modele,
            vecteur_sql,
        ),
    ).fetchone()

identifiant, document, page, numero_figure, dimension = resultat
print(f"Figure indexée : {identifiant}")
print(f"Source : {document}, page {page}, figure {numero_figure}")
print(f"Dimension vérifiée dans PostgreSQL : {dimension}")