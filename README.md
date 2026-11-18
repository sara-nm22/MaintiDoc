# MaintiDoc

Prototype de la Partie I réalisé par Sara Naim à l’EMSI. MaintiDoc répond à des questions sur un PDF et affiche les passages transmis au modèle avec leur page d’origine.

## Fonctionnement

- PyMuPDF extrait le texte des pages. LlamaIndex le découpe en passages.
- E5 multilingue encode les textes, les lignes du tableau de démonstration et une description de figure en vecteurs de 384 valeurs.
- PostgreSQL avec pgvector stocke les passages et leurs métadonnées. CLIP utilise une table distincte pour les vecteurs de figures de 512 valeurs.
- Streamlit permet de charger un PDF, de l’indexer et de consulter les sources. Qwen 2.5 1.5B, exécuté avec Ollama, génère une réponse à partir des extraits. LangChain expose les outils documentaires et visuels, avec un routage de secours en Python.

## Installation sous Windows

Prérequis : Python 3.11, Docker Desktop avec moteur démarré et Ollama. Ouvrir PowerShell à la racine du projet.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install streamlit pymupdf llama-index-readers-file sentence-transformers pillow "psycopg[binary]" langchain-ollama
```

### PostgreSQL et pgvector

Pour une première installation, choisir un mot de passe PostgreSQL quand PowerShell le demande. Exécuter ces commandes dans le même terminal.

```powershell
$env:POSTGRES_PASSWORD = Read-Host "Mot de passe PostgreSQL"
docker run -d --name maintidoc-db -e POSTGRES_DB=maintidoc -e POSTGRES_USER=maintidoc -e POSTGRES_PASSWORD -p 127.0.0.1:5433:5432 -v maintidoc_pgdata:/var/lib/postgresql/data pgvector/pgvector:pg17
Remove-Item Env:POSTGRES_PASSWORD
```

Si le conteneur existe déjà et est arrêté, utiliser `docker start maintidoc-db`. Attendre le message `accepting connections` avant d’activer l’extension.

```powershell
docker exec maintidoc-db pg_isready -U maintidoc -d maintidoc
docker exec maintidoc-db psql -U maintidoc -d maintidoc -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Ollama

Démarrer Ollama depuis le menu Démarrer ou lancer `ollama serve` dans un terminal si le serveur est arrêté. Le laisser disponible sur le port 11434 et télécharger le modèle dans un autre terminal.

```powershell
ollama pull qwen2.5:1.5b
```

### Reconstruire l’index et lancer l’application

Placer le PDF de démonstration [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) dans `data/attention.pdf`. Les scripts d’extraction ciblent la mise en page de l’exemplaire utilisé en cours.

```powershell
.\.venv\Scripts\python.exe .\src\enregistrer_pages.py
.\.venv\Scripts\python.exe .\src\decouper_textes.py
.\.venv\Scripts\python.exe .\src\extraire_tableau.py
.\.venv\Scripts\python.exe .\src\generer_embeddings.py
.\.venv\Scripts\python.exe .\src\importer_embeddings.py
.\.venv\Scripts\python.exe .\src\extraire_schema.py
.\.venv\Scripts\python.exe .\src\indexer_figure.py
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Le premier lancement des encodeurs télécharge les modèles. Les scripts d’import demandent le mot de passe PostgreSQL choisi à la création du conteneur. Dans Streamlit, ouvrir l’URL locale affichée par le terminal et saisir le même mot de passe.

## Démonstration et résultats

| Demande | Résultat observé |
|---|---|
| Affiche-moi le schéma de l’architecture du Transformer. | Figure 1, page 3, retrouvée avec CLIP via le routage Python. |
| Combien de têtes d’attention le Transformer de base utilise-t-il ? | 8 têtes, document de démo, page 5. |
| Selon le Tableau 2, quel est le score BLEU anglais-français du modèle de base ? | 38,1, document de démo, page 8. |
| Dans un PDF importé, poser une question sur un libellé précis entre guillemets. | Sur le rapport PlantScan, le libellé « Stockage local » a permis de retrouver AsyncStorage à la 18e page du PDF. |

Pour tester un autre document, sélectionner un PDF autorisé, cliquer sur **Indexer le PDF**, puis poser une question sur son contenu. Retirer le PDF sélectionné pour revenir à la démo.

## Limites observées

L’import prend en charge le texte sélectionnable. La figure, sa description et les deux lignes du Tableau 2 ont été préparées spécialement pour le PDF de démonstration. La question sur le nombre de têtes utilise une règle propre à cette démo. Le seuil de similarité de 0,78 est provisoire. Le routage de secours a fonctionné lors de la démo, tandis que le choix autonome par Qwen reste à valider. Les extraits affichés permettent de vérifier les réponses générées.
