import json
from getpass import getpass
from urllib.request import Request, urlopen

import psycopg
from sentence_transformers import SentenceTransformer


question = input("Ta question : ").strip()

if not question:
    raise SystemExit("Saisis une question pour lancer la recherche.")

nom_modele = "intfloat/multilingual-e5-small"
modele = SentenceTransformer(nom_modele, device="cpu")

requete_recherche = question
question_normalisee = question.casefold().replace("’", "'")

if "combien de têtes d'attention" in question_normalisee:
    requete_recherche = (
        "Dans la section « Multi-Head Attention », combien de têtes "
        "d’attention parallèles le Transformer de base utilise-t-il ? "
        "Donne la valeur de h."
    )

texte_question = "query: " + requete_recherche
vecteur = modele.encode(texte_question, normalize_embeddings=True)
vecteur_sql = json.dumps(vecteur.tolist(), allow_nan=False)

mot_de_passe = getpass("Mot de passe PostgreSQL : ")
seuil_similarite = 0.78

identifiants_tableau = (
    "attention_tableau_2_ligne_01",
    "attention_tableau_2_ligne_02",
)

with psycopg.connect(
    host="127.0.0.1",
    port=5433,
    dbname="maintidoc",
    user="maintidoc",
    password=mot_de_passe,
    connect_timeout=5,
) as connexion:
    with connexion.cursor() as curseur:
        curseur.execute(
            """
            SELECT id, document, page, texte,
                   embedding <=> %s::vector AS distance
            FROM passages
            WHERE modele = %s
            ORDER BY distance, id
            LIMIT 10
            """,
            (vecteur_sql, nom_modele),
        )
        resultats = curseur.fetchall()

        if not resultats:
            raise SystemExit("Aucun passage trouvé pour ce modèle.")

        identifiant_source, _, _, _, distance_source = resultats[0]
        meilleure_similarite = 1 - distance_source

        if meilleure_similarite < seuil_similarite:
            print("\nRéponse de MaintiDoc")
            print("Je n’ai pas trouvé de passage suffisamment pertinent dans le document.")
            print(f"Meilleur score : {meilleure_similarite:.3f}")
            raise SystemExit(0)

        sources_transmises = [resultats[0][:4]]

        if identifiant_source in identifiants_tableau:
            curseur.execute(
                """
                SELECT id, document, page, texte
                FROM passages
                WHERE modele = %s
                  AND type = 'tableau'
                  AND id IN (%s, %s)
                ORDER BY id
                """,
                (nom_modele, *identifiants_tableau),
            )
            sources_transmises = curseur.fetchall()

            if len(sources_transmises) != len(identifiants_tableau):
                raise SystemExit(
                    "Le Tableau 2 est incomplet en base. "
                    "Relance generer_embeddings.py, puis importer_embeddings.py."
                )

extraits = []

for identifiant, document, page, texte in sources_transmises:
    reference = f"[{document}, p. {page}]"
    extraits.append(f"{reference} | {identifiant}\n{texte}")

contexte = "\n\n".join(extraits)

donnees_ollama = {
    "model": "qwen2.5:1.5b",
    "stream": False,
    "options": {"temperature": 0.1},
    "messages": [
        {
            "role": "system",
            "content": (
                "Réponds directement à la question en français. "
                "Utilise uniquement les informations des extraits. "
                "Pour un score BLEU, identifie d'abord la ligne du modèle demandé. "
                "Lis ensuite le score de la paire de langues dans cette même ligne. "
                "Transformer (base model) est le modèle de base. "
                "Transformer (big) est le grand modèle. "
                "Si une formule répond à la question, décris ses opérations dans l'ordre. "
                "Si les extraits ne suffisent pas, indique-le."
            ),
        },
        {
            "role": "user",
            "content": f"Question : {question}\n\nExtraits du document :\n{contexte}",
        },
    ],
}

requete = Request(
    "http://127.0.0.1:11434/api/chat",
    data=json.dumps(donnees_ollama).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

print("\nQwen prépare la réponse...", flush=True)

with urlopen(requete, timeout=180) as reponse_http:
    resultat_ollama = json.load(reponse_http)

reponse = resultat_ollama["message"]["content"].strip()

print("\nRéponse de MaintiDoc")
print(reponse)

print("\nSources transmises à Qwen")
for identifiant, document, page, _ in sources_transmises:
    print(f"- {identifiant} | [{document}, p. {page}]")

print("\nPassages retrouvés par PostgreSQL")
for identifiant, document, page, _, distance in resultats:
    similarite = 1 - distance
    print(f"- {identifiant} | {document}, p. {page} | score {similarite:.3f}")