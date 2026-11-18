from pathlib import Path

from PIL import Image
from sentence_transformers import SentenceTransformer

racine_projet = Path(__file__).resolve().parents[1]
chemin_image = (
    racine_projet / "data" / "images" / "attention_page_3_figure_1.png"
)

if not chemin_image.is_file():
    raise FileNotFoundError(f"Schéma introuvable : {chemin_image}")

nom_modele = "sentence-transformers/clip-ViT-B-32"
print(f"Chargement du modèle : {nom_modele}", flush=True)
modele = SentenceTransformer(nom_modele, device="cpu")

descriptions = [
    "A diagram of the Transformer encoder and decoder architecture",
    "A photograph of a centrifugal water pump",
]

with Image.open(chemin_image) as fichier_image:
    image = fichier_image.convert("RGB")

vecteur_image = modele.encode([image], normalize_embeddings=True)
vecteurs_textes = modele.encode(descriptions, normalize_embeddings=True)
scores = modele.similarity(vecteur_image, vecteurs_textes)[0]

print("\nSource : attention.pdf, page 3, figure 1")
print(f"Dimension du vecteur image : {vecteur_image.shape[1]}")

for description, score in zip(descriptions, scores, strict=True):
    print(f"{score.item():.3f} | {description}")