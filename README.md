# Chatbot des villes du monde

Un chatbot en **français** qui répond à des questions sur 28 villes (ex. : *« Quelle est la population d'Ottawa ? »*, *« Qui a conçu les bâtiments de Brasilia ? »*).
Ses réponses viennent de pages Wikipédia, pas de sa mémoire. Il tourne sur votre ordinateur, sans clé d'API payante. Seule la météo en direct a besoin d'internet.

## Comment ça marche

Le principe s'appelle **RAG** : le programme cherche d'abord les bons paragraphes, puis demande à un modèle de langage de rédiger la réponse avec eux.

```
PDF Wikipédia → nettoyage → découpage en morceaux → base de données (ChromaDB)
                                                              ↓
Question → quelle ville ? → recherche des morceaux → classement → modèle Qwen → réponse
```

1. **Nettoyage** (`clean.py`) : on extrait le texte des PDF et on enlève les références, liens et parasites.
2. **Indexation** (`indexation.py`) : le texte est découpé en morceaux de 500 caractères et rangé dans une base qui permet de chercher par le sens.
3. **Question** (`app.py`) : le programme repère la ville citée, retrouve 20 morceaux, en garde les 4 meilleurs, puis le modèle `Qwen2.5-1.5B-Instruct` écrit la réponse.

## Villes couvertes

Abuja, Addis-Abeba, Barcelone, Berlin, Brasilia, Buenos Aires, Canberra, Casablanca, Le Caire, Londres, Lyon, Madrid, Marrakech, Marseille, Mexico, New Delhi, New York, Ottawa, Paris, Pékin, Pretoria, Rabat, Riyad, Singapour, Sydney, Tokyo, Washington, Wellington.

Pour en ajouter une : mettre son PDF dans `data/` (ex. `Lisbonne.pdf`), puis relancer les étapes 2 et 3 ci-dessous.

## Installation

Il faut Python 3.10 ou plus récent. Le premier lancement télécharge plusieurs Go de modèles. Une carte graphique accélère les réponses, mais ce n'est pas obligatoire.

```bash
git clone <url-du-depot>
cd <nom-du-depot>
python -m venv venv
source venv/bin/activate        # Windows : venv\\Scripts\\activate
pip install -r requirements.txt
```

## Utilisation

Lancez ces commandes dans l'ordre :

```bash
python clean.py        # 1. PDF de data/ → fichiers \*\_nettoye.txt
python indexation.py   # 2. construit la base (efface et recrée chroma\_db/)
python app.py          # 3. ouvre le chatbot sur http://127.0.0.1:7860
```

Citez la ville dans votre question. Sans ville, le chatbot cherche dans toutes les villes, avec moins de précision. La liste déroulante à gauche ne sert qu'à afficher la météo.

Chaque question est notée dans `logs.csv`.

## Mesurer la qualité

```bash
python evaluate.py              # pose les questions de test\_questions.json (lent)
python evaluate.py --retrieval  # teste seulement la recherche (rapide)
python evaluate.py --ville Ottawa
python analyse.py               # crée les graphiques dans docs/img/
```

`test\_questions.json` contient 179 questions avec les mots-clés attendus. Une question est juste si sa réponse contient un de ses mots-clés. Si le mot-clé n'existe pas dans le fichier de la ville, la question est ignorée et signalée.

## Résultats

Sur 159 questions valides (25 villes) :

||Résultat|
|-|-|
|Le bon passage est retrouvé|91 %|
|La réponse est juste|82 % (marge d'erreur : environ ± 6 points)|
|Durée moyenne d'une réponse|9,5 s|

!\[Où se perdent les erreurs](docs/img/01\_origine\_des\_erreurs.png)

Les erreurs viennent à parts égales de la recherche (12 cas) et du modèle (17 cas). Le défaut le plus gênant : quand il ne trouve pas l'information, le modèle l'**invente** presque toujours au lieu de dire qu'il ne sait pas. Détails et graphiques : [ANALYSE.md](ANALYSE.md).

## Limites

* Les résultats ne portent que sur 25 villes sur 28 (Le Caire, New Delhi et New York n'ont pas pu être évaluées).
* Avec 4 à 9 questions par ville, on ne peut pas comparer les villes entre elles.
* Le test par mots-clés est strict : une bonne réponse reformulée peut être comptée fausse.
* Le mode « sans ville citée » n'est pas encore testé.
* Le modèle (1,5 milliard de paramètres) se trompe parfois sur les dates et les chiffres.

## Fichiers

```text
config.py               réglages communs (chemins, modèle d'embedding)
clean.py                PDF → texte propre
indexation.py           texte → base ChromaDB
app.py                  chatbot (recherche + modèle + interface + météo + logs)
evaluate.py             test automatique avec test\_questions.json
analyse.py              graphiques à partir de evaluation\_resultats.csv
test\_questions.json     179 questions de test
requirements.txt        bibliothèques Python à installer
data/                   PDF et textes \*\_nettoye.txt
chroma\_db/              base de données (créée par indexation.py)
docs/img/               graphiques
```

## Sources et licence

Les textes viennent de Wikipédia (licence CC BY-SA) 

Auteur : S. Younes

