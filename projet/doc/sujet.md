# Projet de gestion des données

## Objectifs

Le but de ce projet est de développer un logiciel pour reconstruire une base de données structurée SQL à partir d'un texte en langage naturel.

Le texte en question se trouve dans le fichier `story.md`.

Il s'agit de procéder de la façon suivante. Il faut d'abord construire un graphe de connaissances à partir du texte en langage naturel. Ce graphe contiendra des triplets de la forme `entité -[relation]-> entité`. Ce graphe sera converti en base Neo4j.

Au moyens de requêtes Cypher, la base de données Neo4j sera dans un premier temps convertie en base noSQL MongoDB.

Enfin, une base de données SQL sera construite à partir de la base de données MongoDB au moyen de requêtes MongoDB.

L'usage de l'IA est autorisée. Vous devez expliquer comment vous avez utilisé l'IA. Vous devez mettre en annexe de votre rapport les requêtes  et les réponses utilisées.

## Questions

### Découpage du texte

Pour développer, on utilisera le mini texte `story.md`. Mais le logiciel développé doit être capable de traiter un texte beaucoup plus long.

La première étape consiste donc à découper le texte en plusieurs fichiers, un par paragraphe, nommées `story_01.md`, `story_02.md`, etc.

Écrire un script Python, `chunking.py`, qui effectue ce découpage.

Par exemple
```python
def chunk_text(file_path, output_folder):
    ...
```

### Traitement du texte

Pour générer le graphe de connaissance, on utilisera un petit LLM (come Mistral Nemo 12B, Qwen3 4B, etc.). Le modèle peut-être installé en local sur votre ordinateur, via llama.cpp ou ollama ou vous pouvez utiliser un service en ligne avec une clé d'API (gratuite).

Écrire un module Python `llm.py` qui permet d'interroger le LLM. La fonction principale de ce module sera `query_llm(question)` qui retournera la réponse du LLM à une question posée.

Ce module devra pouvoir gérer le changement de modèle, de clé d'API et les erreurs (connexion, timeout, taille de contexte, etc.).

### Construction du graphe

Écrire un module Python `kgraph.py` (comme *knowledge graph*) utilisant les modules précédents pour construire un graphe de connaissances à partir du texte en langage naturel. Pour gérer le graphe, on utilisera le module NetworkX et pour sa visualisation on utilisera le module PyVis.

La fonction principale de ce module sera `build_graph(folder)` qui retournera le graphe construit à partir du texte du dossier folder.

Pour cela vous devez concevoir un prompt qui à partir de chaque paragraphe du texte découpé dans le dossier folder, génère un ensemble de triplets de la forme `entité -[relation]-> entité`.
Il faut ensuite parser ces triplets pour les ajouter au graphe.

Points importants:

- utiliser des blocs try/except pour gérer les erreurs, inévitables avec l'aspect aléatoire des LLM.
- utiliser la visualisation PyVis pour déboguer votre code de façon proactive.

### Nettoyage du graphe

Le graphe construit peut contenir des entités et des relations qui ne sont pas pertinentes. Il faut donc nettoyer le graphe.

Pour cela utiliser un modèle de similarité pour fusionner les entités et relations redondantes. Par exemple "étudiant", "etudiant", "étudiante" deviennent "étudiant". Ou "chat", "chatte", "chaton" deviennent "chat".

Utiliser des modèles de la bibliothèque PyTorch.

On pourra tester plusieurs alternatives: `paraphrase-multilingual-MiniLM-L12-v2`, 
`intfloat/multilingual-e5-small`, `dangvantuan/sentence-camembert-base``

Justifer vos choix en présentant les avantages et les inconvénients de chaque modèle et les résultats de vos expériences.

### Conversion en base Neo4j

Écrire un module Python `cypher.py` qui permet de convertir le graphe en base Neo4j. Ce module doit générer les commandes Cypher pour créer les noeuds et les relations.

Ce module ne doit pas appeler un LLM pour garantir le déterminisme et le succès de cette conversion !

### Conversion en base MongoDB

Écrire un module Python `cypher2mongo.py` qui permet de convertir la base Neo4j en base MongoDB. Indication: 

### Conversion en base SQL

Écrire un module Python `mongo2sql.py` qui permet de convertir la base MongoDB en base SQL.

### Pour aller plus loin

Quelques suggestions pour améliorer les capacités de votre outil:

- préparer un ensemble de tests avec des documents plus longs et des cas d'usage plus complexes. Ces tests doivent pouvir être contrôlés à la main (donc ni trop longs ni trop courts)

- préparer un cas d'usage utilisant un long document issu d'un problème réel. 

- dans le module de nettoyage du graphe, vous pouvez tester des approches par cross-encoder. Vous pouvez tester par exemple avec des modèles comme `cross-encoder/multilingual-mpnet-base-v2` ou `cross-encoder/multilingual-mpnet-base-v2`



