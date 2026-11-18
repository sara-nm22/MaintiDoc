import json
import unicodedata
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

racine_projet = Path(__file__).resolve().parents[1]
dossier_donnees = racine_projet / "data" / "processed"
chemin_pages = dossier_donnees / "attention_pages.json"
chemin_passages = dossier_donnees / "attention_passages.json"

with chemin_pages.open("r", encoding="utf-8") as fichier:
    pages = json.load(fichier)

documents = []

for page in pages:
    texte = unicodedata.normalize("NFKC", page["texte"])

    # Corrections des caractères identifiés dans ce PDF.
    texte = texte.replace(chr(57420), "fi")
    texte = texte.replace(chr(57421), "fl")
    texte = texte.replace(chr(57346), "ε")

    document = Document(
        text=texte,
        metadata={
            "document": page["document"],
            "page": page["page"],
            "type": page["type"],
        },
    )
    documents.append(document)

decoupeur = SentenceSplitter(
    chunk_size=400,
    chunk_overlap=60,
)
passages = decoupeur.get_nodes_from_documents(documents)

passages_exportes = []

for numero, passage in enumerate(passages, start=1):
    donnees_passage = {
        "id": f"attention_passage_{numero:04d}",
        "document": passage.metadata["document"],
        "page": passage.metadata["page"],
        "type": passage.metadata["type"],
        "texte": passage.text,
    }
    passages_exportes.append(donnees_passage)

caracteres_restants = 0

for passage in passages_exportes:
    for code in (57420, 57421, 57346):
        caracteres_restants += passage["texte"].count(chr(code))

with chemin_passages.open("w", encoding="utf-8") as fichier:
    json.dump(
        passages_exportes,
        fichier,
        ensure_ascii=False,
        indent=2,
    )

print(f"Pages chargées : {len(pages)}")
print(f"Passages créés : {len(passages_exportes)}")
print(f"Caractères ciblés restants : {caracteres_restants}")
print(f"Fichier créé : {chemin_passages}")