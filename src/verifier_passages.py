import json
from pathlib import Path

racine_projet = Path(__file__).resolve().parents[1]
chemin_json = racine_projet / "data" / "processed" / "attention_passages.json"

with chemin_json.open("r", encoding="utf-8") as fichier:
    passages = json.load(fichier)

print(f"Nombre total de passages : {len(passages)}")

for passage in passages[:3]:
    print("\n" + "-" * 50)
    print(f"Identifiant : {passage['id']}")
    print(f"Source : {passage['document']}, page {passage['page']}")
    print(passage["texte"])