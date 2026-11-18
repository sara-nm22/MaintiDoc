import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


racine_projet = Path(__file__).resolve().parents[1]
dossier_donnees = racine_projet / "data" / "processed"

chemin_passages = dossier_donnees / "attention_passages.json"
chemin_tableau = dossier_donnees / "attention_tableau_2.json"
chemin_embeddings = dossier_donnees / "attention_embeddings.json"

with chemin_passages.open("r", encoding="utf-8") as fichier:
    passages = json.load(fichier)

with chemin_tableau.open("r", encoding="utf-8") as fichier:
    tableau = json.load(fichier)

numero_tableau = tableau["numero_tableau"]

for numero_ligne, ligne in enumerate(tableau["lignes"], start=1):
    texte_tableau = (
        f"Tableau {numero_tableau}, WMT 2014. "
        f"Modèle : {ligne['modele']}. "
        f"Score BLEU anglais-allemand (EN-DE) : {ligne['bleu_en_de']}. "
        f"Score BLEU anglais-français (EN-FR) : {ligne['bleu_en_fr']}."
    )

    passages.append({
        "id": f"attention_tableau_{numero_tableau}_ligne_{numero_ligne:02d}",
        "document": tableau["document"],
        "page": tableau["page"],
        "type": "tableau",
        "texte": texte_tableau,
    })

# Ce bloc est en dehors de la boucle du tableau.
passages.append({
    "id": "attention_figure_01_description",
    "document": "attention.pdf",
    "page": 3,
    "type": "description_figure",
    "texte": (
        "Figure 1 — Schéma de l’architecture du Transformer. "
        "À gauche, l’encodeur traite les entrées avec des blocs "
        "d’attention multi-têtes et des réseaux feed-forward. "
        "À droite, le décodeur utilise une attention masquée sur "
        "les sorties précédentes et une attention sur les sorties de l’encodeur. "
        "Une couche linéaire suivie d’un softmax produit les probabilités de sortie."
    ),
})

nom_modele = "intfloat/multilingual-e5-small"
print(f"Chargement du modèle : {nom_modele}", flush=True)
modele = SentenceTransformer(nom_modele, device="cpu")

textes = []

for passage in passages:
    texte = "passage: " + passage["texte"]
    tokens = modele.tokenizer(texte, truncation=False, verbose=False)
    nombre_tokens = len(tokens["input_ids"])

    if nombre_tokens > modele.max_seq_length:
        raise ValueError(
            f"{passage['id']} contient {nombre_tokens} tokens. "
            f"Limite du modèle : {modele.max_seq_length}."
        )

    textes.append(texte)

vecteurs = modele.encode(
    textes,
    batch_size=8,
    normalize_embeddings=True,
    show_progress_bar=True,
)

for passage, vecteur in zip(passages, vecteurs, strict=True):
    passage["embedding"] = vecteur.tolist()

resultat = {
    "modele": nom_modele,
    "dimension": int(vecteurs.shape[1]),
    "passages": passages,
}

with chemin_embeddings.open("w", encoding="utf-8") as fichier:
    json.dump(resultat, fichier, ensure_ascii=False, allow_nan=False)

print(f"Passages encodés : {len(passages)}")
print(f"Dimension des embeddings : {resultat['dimension']}")
print(f"Fichier créé : {chemin_embeddings}")