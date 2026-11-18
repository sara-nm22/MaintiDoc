from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def rechercher_document(requete: str) -> str:
    """Recherche des informations dans le texte et les tableaux des PDF."""
    return f"Test de recherche documentaire : {requete}"


@tool
def rechercher_figure(requete: str) -> str:
    """Retrouve une image ou un schéma du PDF pour l'afficher à l'utilisateur."""
    return f"Test de recherche de figure : {requete}"


outils = {
    rechercher_document.name: rechercher_document,
    rechercher_figure.name: rechercher_figure,
}

modele = ChatOllama(
    model="qwen2.5:1.5b",
    base_url="http://127.0.0.1:11434",
    temperature=0,
)

modele_avec_outils = modele.bind_tools(list(outils.values()))

question = input("Ta demande : ").strip()

if not question:
    raise SystemExit("Saisis une demande pour lancer le test.")

reponse = modele_avec_outils.invoke([
    (
        "system",
        "Choisis exactement un outil. "
        "Utilise rechercher_figure si la personne demande à voir une image ou un schéma. "
        "Utilise rechercher_document si elle demande une information, "
        "une explication ou un chiffre. Transmets sa demande dans le champ requete.",
    ),
    ("human", question),
])

if reponse.tool_calls:
    appel = reponse.tool_calls[0]
    nom_outil = appel["name"]
    arguments = appel["args"]
    origine = "Qwen"
else:
    demande = question.casefold()
    demande_affichage = any(
        mot in demande for mot in ("affiche", "montre", "voir")
    )
    demande_figure = any(
        mot in demande for mot in ("schéma", "figure", "image")
    )

    if demande_affichage and demande_figure:
        nom_outil = rechercher_figure.name
    else:
        nom_outil = rechercher_document.name

    arguments = {"requete": question}
    origine = "routage de secours Python"

print(f"Outil sélectionné : {nom_outil}")
print(f"Origine du choix : {origine}")
print(f"Arguments : {arguments}")

if nom_outil in outils:
    resultat = outils[nom_outil].invoke(arguments)
    print(f"Résultat : {resultat}")