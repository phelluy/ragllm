#import "@preview/typslides:1.2.6": *

#show: typslides.with(
  ratio: "16-9",
  theme: "bluey",
  font: "Fira Sans",
)

#front-slide(
  title: "Gestion de Données, LLM et RAG",
  subtitle: "De la donnée structurée à l'IA générative",
  authors: "Philippe Helluy",
)

#slide(title: "Plan du cours")[
  #table-of-contents()
]

// ==========================================
// PARTIE 1 : Bases de Données
// ==========================================

#title-slide[
  Bases de Données SQL
]

// --- Section 1.1 : SQL ---
#slide(title: "Bases de Données Relationnelles (SQL)")[
  *Structure en Tables*
  - Les données sont organisées en *tables* (lignes et colonnes).
  - Schéma rigide : chaque colonne a un type défini.
  - Relations gérées par des *clés* (Primaires / Étrangères).

  *Exemple: base de données d'un vétérinaire*
  - Table `Client` (id, nom, prenom, profession)
  - Table `Animal` (id, espece, age, maitre_id)

  Logiciels: Oracle, PostgreSQL, MySQL, SQLite, _etc._
]

#slide(title: "SQL Avancé : Les Jointures")[
  *Reconstituer l'information dispersée*

  Exemple : Lister les animaux et leurs maîtres.
  ```sql
  SELECT Animal.nom, Client.nom
  FROM Animal
  JOIN Client ON Animal.maitre_id = Client.id
  WHERE Animal.espece = 'Chien';
  ```
  - *Puissant* pour les rapports structurés.
  - *Lent* si beaucoup de données (complexité $O(N*M)$).
]

#slide(title: "SQL et Python")[
  La bibliothèque standard `sqlite3` permet d'interagir facilement.

  ```python
  import sqlite3
  conn = sqlite3.connect('vétérinaire.db')
  cursor = conn.cursor()

  cursor.execute("SELECT * FROM Animal")
  rows = cursor.fetchall()
  for row in rows:
      print(row) # (1, 'Rex', 'Chien', 1)
  ```
  *Pandas* est aussi très utilisé : `df = pd.read_sql(...)`.


]

#slide(title: "Limites du SQL pour l'IA moderne")[
  - *Rigidité* : Difficile de changer le schéma en cours de route.
  - *Jointures coûteuses* : Reconstituer des objets complexes nécessite de lourdes opérations.
  - *Données non structurées* : Mal adapté pour stocker du texte brut volumineux ou des vecteurs (bien que pgvector existe).
]

// --- Section 1.2 : NoSQL Document (MongoDB) ---
#title-slide[
  NoSQL: documents et graphes
]

#slide(title: "Approche Document")[
  - *Format JSON* : Données stockées sous forme de documents hiérarchiques.
  - *Schéma Flexible* : Chaque document peut avoir une structure différente.
  - *Dénormalisation* : On stocke les données liées ensemble (ex: les animaux DANS l'objet client).
  *Avantages* :
  - Récupération rapide des objets complets (pas de jointure)
  - Flexibilité pour l'évolution du schéma
  - Performance en lecture
]

#slide(title: "Exemple JSON (vétérinaire)")[
  ```json
  {
    "id": 1,
    "nom": "Dupont",
    "prenom": "Jean",
    "animaux": [
      { "nom": "Rex", "espece": "Chien" },
      { "nom": "Miaou", "espece": "Chat" }
    ]
  }
  ```
  *Avantage* : plus souple si on veut ajouter des items (la nationalité, par exemple)
]

#slide(title: "Requêtes MongoDB (MQL)")[
  Syntaxe basée sur des dictionnaires JSON.

  *Trouver les clients qui ont un Chien :*
  ```javascript
  db.clients.find({
    "animaux.espece": "Chien"
  })
  ```

  *Agrégation (Compter les animaux par espèce) :*
  ```javascript
  db.clients.aggregate([
    { $unwind: "$animaux" },
    { $group: { _id: "$animaux.espece", count: { $sum: 1 } } }
  ])
  ```
]

#slide(title: "Python et MongoDB (PyMongo)")[
  Librairie `pymongo` pour l'interaction.

  ```python
  from pymongo import MongoClient
  client = MongoClient('localhost', 27017)
  db = client.veterinaire

  # Insertion
  db.clients.insert_one({"nom": "Martin", "animaux": []})

  # Recherche
  for doc in db.clients.find({"nom": "Dupont"}):
      print(doc['prenom'])
  ```
]

#slide(title: "Comparaison SQL vs Document")[
  #table(
    columns: (1fr, 1fr),
    inset: 10pt,
    [*SQL (Relationnel)*], [*NoSQL (Document)*],
    [Tables plates], [Documents imbriqués],
    [Schéma rigide], [Schéma flexible],
    [Idéal pour transactions (Banque)], [Idéal pour le Web / Contenu],
    [Jointures complexes], [Données auto-contenues],
  )
]

#slide(title: "Pourquoi le Graphe ?")[
  - *Le monde est interconnecté*.
  - Les relations sont aussi importantes que les données elles-mêmes.
  - En SQL, les relations sont des contraintes techniques (clés).
  - En Graphe, les relations sont des *objets de première classe*.
]

#slide(title: "Modèle de Données Graphe")[
  - Parfois, les relations sont aussi importantes que les données elles-mêmes.
  - Dans le modèle SQL, les relations (tables) sont souvent peu nombreuses par rapport aux données.
  - Dans le modèle Graphe, les relations sont généralement aussi nombreuses que les données.
  - *Nœuds (Nodes)* : Les entités (ex: `Personne`, `Animal`).
  - *Relations (Edges)* : Les liens (ex: `POSSEDE`, `AIME`, `EST_AMIS_AVEC`).
  - *Propriétés* : Clé-valeur sur les nœuds ET les relations.

  #align(center)[
    `(Jean:Personne) -[:POSSEDE]-> (Rex:Animal)`
  ]
]

#slide(title: "Le Langage Cypher")[
  - Lagngage SQL: requêtes sur des tables, *Cypher*: requête sur des graphes.
  - Syntaxe visuelle (ASCII-Art).
  - Requêtes basées sur le filtrage par motif (_pattern matching_).

  *Exemple : Qui possède Rex ?*
  ```cypher
  MATCH (p:Personne)-[:POSSEDE]->(a:Animal)
  WHERE a.nom = 'Rex'
  RETURN p.nom
  ```
]

#slide(title: "Cypher Avancé : Chemins")[
  Trouver les connexions indirectes (Amis d'amis).

  ```cypher
  MATCH (p:Personne)-[:AMI_AVEC]->(ami)-[:AMI_AVEC]->(p2:Personne)
  RETURN p.nom, p2.nom
  ```

  *Chemin le plus court :*
  ```cypher
  MATCH path = shortestPath(
      (p:Personne {nom:'Jean'})-[*]-(p:Personne {nom:'Claire'})
  )
  RETURN path
  ```
]

#slide(title: "Python et Neo4j")[
  Utilisation du driver officiel `neo4j` ou `py2neo`.

  ```python
  from neo4j import GraphDatabase
  driver = GraphDatabase.driver("bolt://localhost:7687")

  def get_friends(tx, name):
      result = tx.run("MATCH (a:Person)-[:AMI]->(b) "
                      "WHERE a.name = $name RETURN b.name", name=name)
      return [record["b.name"] for record in result]

  with driver.session() as session:
      friends = session.execute_read(get_friends, "Jean")
  driver.close()
  ```
]


#title-slide[
  RAG - Retrieval Augmented Generation
]

#slide(title: "Pourquoi le RAG (Retrieval Augmented Generation) ?")[
  Les LLM sont impressionnants mais :
  1. *Connaissances figées* : Ils ne savent rien après leur date d'entraînement.
  2. *Hallucinations* : Ils inventent des faits s'ils ne savent pas, ne citent pas leurs sources ou en inventent de nouvelles.
  3. *Données limitées* : Ils ne connaissent pas les documents non publics.
  4. *Contexte borné* : Les LLM traitent un contexte contraint en taille.

  - Le RAG  consiste à extraire d'une base de données (qui peut être énorme) le contexte sourcé et pertinent pour la question posée, en respectant la contrainte de taille.
  - Le choix du système de requêtes (texte, embedding, SQL, JSON, graphe, _etc._) impacte la qualité des réponses de l'IA.
]

// ==========================================
// PARTIE 2 : RAG et LLM
// ==========================================


#slide(title: "Architecture RAG vectoriel")[
  *Principe* : Utiliser un petit réseau de neurones pour extraire du sens dans la couche cachée d'embeddings (base de données vectorielle).

  1. *Indexation (Offline)* :
    #align(center)[`Documentation -> Chunking -> Embedding -> Base de données Vectorielle`]

  2. *Recherche (Retrieval)* :
    - Question utilisateur $q$ $->$ Embedding $E_(w)(q)$.
    - Recherche des $k$ vecteurs les plus proches (Similarité Cosinus).

  3. *Génération (Augmentation)* :
    - On insère les $k$ chunks correspondants aux vecteurs trouvés dans le prompt.
    - Le LLM génère la réponse en utilisant ces infos.
]

#slide(title: "Embeddings : Définition Formelle")[
  Soit  $cal(T)$ l'espace des textes, c'est à dire $RR^(N_V times L)$, où $N_V$ est le nombre de tokens dans le vocabulaire et $L$ la longueur des phrases à analyser.
  Un modèle d'embedding est un RN de transformeurs avec des poids $w$:
  $ E_(w) : cal(T) arrow.r RR^D $
  où $D$ est la dimension (ex: 384, 1024).

  *Propriété recherchée* :
  Si deux textes $t_1$ et $t_2$ sont sémantiquement proches, alors leur *similarité cosinus* est proche de 1.

  $ "sim"(t_1, t_2) = cos(E_(w)(t_1), E_(w)(t_2)) = frac(E_(w)(t_1) dot E_(w)(t_2), ||E_(w)(t_1)|| ||E_(w)(t_2)||) $
]

#slide(title: "Entraînement : Contrastive Learning")[
  Comment trouver les poids $w$ ? On utilise des *triplets* de données :
  - *Anchor* ($A$) : Une question (ex: "Capitale de la France ?").
  - *Positive* ($P$) : La bonne réponse (ex: "Paris est la capitale...").
  - *Negative* ($N$) : Une mauvaise réponse (ex: "La pizza est un plat...").

  *Objectif* : Maximiser $"sim"(A, P)$ et minimiser $"sim"(A, N)$.

  On minimise la *Triplet Loss* avec une marge $alpha$ (écart minimal imposé) :
  $ cal(L) = max(0, "sim"(A, N) - "sim"(A, P) + alpha) $
  $=>$ On veut que $ "sim"(A, P) > "sim"(A, N) + alpha $.
]

#slide(title: [Algorithme 1: Ranking (Bi-Encoder)])[
  On dispose d'une base de chunks de documentation $cal(D) = {d_1, ..., d_N}$.

  1. *Indexation (Offline)* :
    $ forall i, v_i = E_(w)(d_i) in RR^D $ (Stockés dans une base vectorielle: on garde le lien entre $d_i$ et $v_i$).

  2. *Recherche (Online)* :
    Pour une requête $q$, on calcule $u = E_(w)(q)$.
    On cherche les $k$ indices $i$ maximisant $"sim"(u, v_i)$.

  3. *Augmentation (Online)* : On ajoute les documents correspondants à ces indices dans le contexte.
]

#slide(title: "Pipeline RAG Implémenté (Voir `rag_demo.py`)")[
  ```python
  # 1. Initialisation & Indexation
  rag = SimpleRAG(data_dir="data")
  rag.load_documents()      # Lecture des chunks (Markdown)
  rag.create_embeddings()   # Calcul des embeddings

  # 2. Retrieval (Recherche)
  query = "Qui est Jules ?"
  results = rag.search(query, top_k=3)
  # results contient les tuples (document, score)

  # 3. Génération (Augmentation)
  context_docs = [doc for doc, score in results]
  response = rag.generate_with_llm(query, context_docs)
  print(response)
  ```
]


#slide(title: "Limites du Bi-Encoder (Vecteurs)")[
  Le modèle vectoriel ("Bi-Encoder") compresse chaque phrase *séparément*. Il perd les nuances syntaxiques croisées.

  *Exemple piège* :
  - *Question* ($q$) : "Qui a mangé la souris ?"
  - *Doc 1* ($d_1$) : "Le python a mangé la souris." (Utile)
  - *Doc 2* ($d_2$) : "Une fois, une souris a mangé un python." (Anecdotique)

  $=>$ Le Bi-Encoder va exprimer une forte similarité entre $d_1$ et $d_2$ (mêmes mots clés).
  $=>$ Nécessité d'un *Cross-Encoder* pour inspecter les deux phrases ensemble et affiner le score de pertinence.
]

#slide(title: "Algorithme 2: Reranking (Cross-Encoder)")[
  L'Encoder compresse tout le sens dans un seul vecteur. Calcul rapide de la similarité, mais pertinence parfois insuffisante.
  Pour plus de précision, on utilise un modèle de *Reranking* $R_(w)$ (réseau de neurones plus gros) qui prend *deux* textes en entrée :

  $ R_(w) : cal(T) times cal(T) arrow.r [0, 1] $
  $ s_(i) = R_(w)(q, d_i) $ (Score de pertinence)

  *Pipeline Complet* :
  1. *Retrieve* : L'Encoder sélectionne $K approx 100$ candidats (rapide).
  2. *Rerank* : Le Cross-Encoder re-calcule un score plus précis pour ces $K$ candidats (lent).
  3. *Top-k* : On garde les $k approx 5$ meilleurs pour le LLM.
]

#slide(title: "Exemple Concret : L'affaire du Collier")[
  *Contexte (9 chunks)* :
  - Un vol de collier a eu lieu au bal du Baron.
  - Sophie sort avec *Jules*, qui porte une *bague serpent*.
  - La *bague serpent* est le symbole du gang "La Griffe".
  - "La Griffe" est un groupe de voleurs d'élite.

  *Question* : "Pourquoi Jules est-il suspect ?"
  - *Oups !* Le RAG standard (top-k=4) échoue souvent.
  - *Pourquoi ?* L'info est éparpillée. Si on récupère "Jules aime Sophie" et "Le Baron vend des rillettes", il manque le lien "Bague" $->$ "Gang".
  - C'est un problème de *Multi-hop Reasoning*.
]

// --- Section 2.2 : RAG Avancé et Graphes ---
#title-slide[
  RAG et graphes
]


#slide(title: "Graph RAG : Enrichissement du contexte")[
  Le graphe de connaissances $G=(V, cal(E))$ structure l'info en *triplets* $(E, V, E')$ :
  *(Entité, Verbe, Entité')*.
  - Ex: `(Jules, PORTE, Bague)`, `(Bague, SYMBOLE, La Griffe)`.

  *Algorithme d'enrichissement (Voir `rag_graph.py`)* :
  1. *Extraction* : Dans les chunks, identifier les entités $e_j$.
  2. *Traversée* : Pour chaque entité $e_j$, explorer le voisinage dans $G$ à une profondeur $p$ (ex: $p=2$).
    $ S_j = { (u, r, v) in cal(E) | "dist"(e_j, u) <= p } $
  3. *Injection* : Transformer les triplets trouvés en texte "Sujet prédicat Objet".
  4. *Prompt* : "Voici le contexte textuel [...] ET voici les relations connues : [...]".
]


#slide(title: "Exemple d'apport du Graphe")[
  *Question* : "Pourquoi Jules est-il suspect ?"

  - *Vectoriel* : Récupère "Jules sort avec Sophie" et "Le Baron a été volé". $=>$ Pas de lien évident.
  - *Graphe* :
    1. Trouve le nœud `Jules`.
    2. Traverse `(Jules)-[:PORTE]->(Bague)`.
    3. Traverse `(Bague)-[:SYMBOLE]->(La Griffe)`.
    4. Traverse `(La Griffe)-[:TYPE]->(Voleurs)`.
    $=>$ Le LLM a accès au chemin (inclus dans le contexte) et déduit que Jules est lié aux voleurs.
]

// --- Section 2.3 : Mise en œuvre ---


#slide(title: "Les outils")[
  Ne pas réinventer la roue :

  - *LangChain* : très populaire, pour assembler des briques LLM et des outils.
  - *LlamaIndex* : spécialisé dans la gestion de données pour LLM (Connecteurs, Index Vectoriels, Index Graphes).
  - *HuggingFace* : incontournable pour les modèles et datasets (dont les encoders).
]

#slide(title: "Exemple avec LlamaIndex (Voir `rag_graph.py`)")[
  #text(size: 16pt)[
    ```python
    from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader

    # 1. Ingestion
    documents = SimpleDirectoryReader("data/").load_data()

    # 2. Construction du Graphe
    # LlamaIndex extrait automatiquement les triplets (E, V, E')
    graph_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=2,
        include_embeddings=True
    )

    # 3. Recherche (utilise le graphe + vecteurs)
    query_engine = graph_index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize"
    )
    response = query_engine.query("Pourquoi Jules est suspect ?")
    ```
  ]
]

#slide(title: "Dernières tendances")[
  - Les LLM ont des contextes de plus en plus grand (1 M tokens).
  - Le RAG a moins le vent en poupe.
  - On utilise de plus en plus des agents pour fouiller dans les données de façon itérative.
  - Les données sont donc stockées sous divers formats à l'extérieur du prompt.
  - On ne pose plus la question à un LLM mais à un agent IA: Google Antigravity, Microsoft Copilot, mistral-vibe, _etc._
]

#slide(title: "Bibliographie")[
  #bibliography("rag.bib", full: true, style: "ieee")
]

