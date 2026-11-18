from pathlib import Path

from llama_index.readers.file import PyMuPDFReader

racine_projet = Path(__file__).resolve().parents[1]
chemin_pdf = racine_projet / "data" / "attention.pdf"

if not chemin_pdf.is_file():
    raise FileNotFoundError(f"PDF introuvable : {chemin_pdf}")

lecteur = PyMuPDFReader()
print(f"Lecteur utilisé : {type(lecteur).__name__}")

pages = lecteur.load_data(file_path=str(chemin_pdf), metadata=True)

print(f"Nombre de pages lues : {len(pages)}")

premiere_page = pages[0]
numero_page = premiere_page.metadata["source"]
print(f"Source : {chemin_pdf.name}, page {numero_page}")

texte = premiere_page.text
position_resume = texte.find("Abstract")

if position_resume != -1:
    print("\nExtrait du résumé :")
    print(texte[position_resume:position_resume + 1000])
else:
    print("\nRésumé introuvable. Début du texte extrait :")
    print(texte[:1500])