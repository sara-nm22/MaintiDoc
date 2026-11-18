import json
from getpass import getpass
from pathlib import Path

import psycopg

racine_projet = Path(__file__).resolve().parents[1]
chemin_json = racine_projet / "data" / "processed" / "attention_embeddings.json"

with chemin_json.open("r", encoding="utf-8") as fichier:
    donnees = json.load(fichier)

if donnees["dimension"] != 384:
    raise ValueError("Le fichier doit contenir des embeddings de dimension 384.")

lignes = []

for passage in donnees["passages"]:
    ligne = (
        passage["id"],
        passage["document"],
        passage["page"],
        passage["type"],
        passage["texte"],
        donnees["modele"],
        json.dumps(passage["embedding"], allow_nan=False),
    )
    lignes.append(ligne)

mot_de_passe = getpass("Mot de passe PostgreSQL : ")

with psycopg.connect(
    host="127.0.0.1",
    port=5433,
    dbname="maintidoc",
    user="maintidoc",
    password=mot_de_passe,
    connect_timeout=5,
) as connexion:
    with connexion.cursor() as curseur:
        curseur.execute("""
            CREATE TABLE IF NOT EXISTS passages (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                page INTEGER NOT NULL,
                type TEXT NOT NULL,
                texte TEXT NOT NULL,
                modele TEXT NOT NULL,
                embedding vector(384) NOT NULL
            )
        """)

        curseur.executemany("""
            INSERT INTO passages (id, document, page, type, texte, modele, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                document = EXCLUDED.document,
                page = EXCLUDED.page,
                type = EXCLUDED.type,
                texte = EXCLUDED.texte,
                modele = EXCLUDED.modele,
                embedding = EXCLUDED.embedding
        """, lignes)

        curseur.execute("""
            SELECT COUNT(*), MIN(vector_dims(embedding)), MAX(vector_dims(embedding))
            FROM passages
        """)
        nombre, dimension_min, dimension_max = curseur.fetchone()

print(f"Passages en base : {nombre}")
print(f"Dimensions min et max : {dimension_min}, {dimension_max}")