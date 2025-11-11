// Import des packages, comme dans le premier exemple
#import "@preview/typslides:1.2.6": *
#import "@preview/sourcerer:0.2.1": code // Optionnel, mais bonne pratique

// Configuration du thème, comme dans le premier exemple
#show: typslides.with(
  ratio: "16-9",
  theme: "bluey",
  font: "Fira Sans",
  /* language: "fr" // francisation du document */
)

// Page de titre, syntaxe typslides (front-slide)
#front-slide(
  title: "Introduction au RAG pour les LLM",
  subtitle: "Retrieval-Augmented Generation",
  authors: "Cours M1",
)

// Diapositives de contenu, syntaxe typslides (slide(title: ...)[...])
#slide(title: "Plan du cours")[
  1. Introduction et motivation
  2. Concepts fondamentaux
  3. Architecture d'un système RAG
  4. Embeddings et similarité
  5. Stockage vectoriel
  6. Récupération de documents
  7. Génération augmentée
  8. Mise en pratique
]

#slide(title: "Qu'est-ce qu'un LLM ?")[
  *Large Language Model (LLM)*
  
  - Modèle de langage pré-entraîné sur d'énormes corpus de texte
  - Capable de générer du texte cohérent
  - Exemples : GPT, LLaMA, Mistral, etc.
  
  *Limitations :*
  - Connaissance figée au moment de l'entraînement
  - Hallucinations (génération d'informations fausses)
  - Pas d'accès à des données spécifiques ou privées
]

#slide(title: "Le problème")[
  *Comment donner accès à des informations à jour ou spécifiques ?*
  
  #v(1em)
  
  Solutions possibles :
  - Ré-entraîner le modèle ? → Coûteux et lent
  - Fine-tuning ? → Limité, risque de catastrophic forgetting
  - *RAG* → Léger, flexible, efficace !
]

#slide(title: "Qu'est-ce que le RAG ?")[
  *Retrieval-Augmented Generation*
  
  Technique qui combine :
  
  1. *Retrieval (Récupération)* : Recherche d'informations pertinentes dans une base de connaissances
  
  2. *Augmentation* : Enrichissement du contexte du LLM
  
  3. *Generation* : Production de réponse par le LLM avec le contexte enrichi
]

#slide(title: "Principe de base du RAG")[
  ```
  Question utilisateur
        ↓
  Récupération de documents pertinents
        ↓
  Construction du prompt avec les documents
        ↓
  Génération par le LLM
        ↓
  Réponse basée sur les documents
  ```
]

#slide(title: "Architecture générale")[
  #table(
    columns: (1fr, 1fr),
    stroke: none,
    [*Phase d'indexation*], [*Phase de requête*],
    [
      1. Documents sources
      2. Découpage en chunks
      3. Génération d'embeddings
      4. Stockage vectoriel
    ],
    [
      1. Question utilisateur
      2. Embedding de la question
      3. Recherche de similarité
      4. Génération de réponse
    ]
  )
]

#slide(title: "Phase 1 : Indexation")[
  *Préparation de la base de connaissances*
  
  1. *Collecte* : Rassembler les documents (PDF, web, bases de données)
  
  2. *Découpage* : Diviser en chunks de taille appropriée
  
  3. *Embedding* : Convertir chaque chunk en vecteur
  
  4. *Stockage* : Sauvegarder dans une base vectorielle
]

#slide(title: "Qu'est-ce qu'un embedding ?")[
  *Représentation vectorielle d'un texte*
  
  - Transforme du texte en vecteur de nombres réels
  - Capture le sens sémantique
  - Textes similaires → vecteurs proches
  
  Exemple :
  - "chat" et "félin" → vecteurs proches
  - "chat" et "voiture" → vecteurs éloignés
]

#slide(title: "Modèles d'embedding")[
  *Exemples de modèles populaires :*
  
  - Sentence-BERT (SBERT)
  - all-MiniLM-L6-v2
  - multilingual-e5-base
  - text-embedding-ada-002 (OpenAI)
  
  *Caractéristiques importantes :*
  - Dimension du vecteur (384, 768, 1536...)
  - Support multilingue
  - Performance vs vitesse
]

#slide(title: "Le chunking")[
  *Découpage des documents en morceaux*
  
  Stratégies :
  - Par taille fixe (ex: 500 tokens)
  - Par paragraphe ou section
  - Avec chevauchement (overlap)
  
  Trade-offs :
  - Chunks trop petits → perte de contexte
  - Chunks trop grands → dilution de l'information
]

#slide(title: "Bases de données vectorielles")[
  *Stockage et recherche de vecteurs*
  
  Solutions populaires :
  - FAISS (Facebook AI)
  - Chroma
  - Pinecone
  - Weaviate
  - Milvus
  
  Permettent une recherche efficace par similarité
]

#slide(title: "Mesures de similarité")[
  *Comment comparer deux vecteurs ?*
  
  1. *Similarité cosinus* : Angle entre vecteurs
     - Valeur entre -1 et 1
     - Indépendant de la magnitude
  
  2. *Distance euclidienne* : Distance géométrique
  
  3. *Produit scalaire* : Dot product
]

#slide(title: "Phase 2 : Récupération")[
  *Recherche de documents pertinents*
  
  1. Recevoir la question de l'utilisateur
  
  2. Générer l'embedding de la question
  
  3. Calculer la similarité avec tous les chunks
  
  4. Sélectionner les k chunks les plus similaires
  
  Généralement : k = 3 à 5 documents
]

#slide(title: "Phase 3 : Génération")[
  *Construction du prompt augmenté*
  
  ```
  Contexte:
  [Document 1 pertinent]
  [Document 2 pertinent]
  [Document 3 pertinent]
  
  Question: [Question de l'utilisateur]
  
  Réponse: 
  ```
  
  Le LLM génère une réponse basée sur le contexte fourni
]

#slide(title: "Avantages du RAG")[
  - ✓ Accès à des informations à jour
  - ✓ Pas besoin de ré-entraînement
  - ✓ Réduction des hallucinations
  - ✓ Sources traçables
  - ✓ Données privées/spécifiques
  - ✓ Moins coûteux que le fine-tuning
  - ✓ Facilement actualisable
]

//#slide(title: "Limites
#slide[
  == Limites et défis
  
  - Qualité dépend de la base de documents
  - Chunking optimal difficile à déterminer
  - Coût computationnel de l'embedding
  - Taille du contexte limitée
  - Besoin d'infrastructure (base vectorielle)
  - Latence de récupération
]

#slide[
  == Variantes et améliorations
  
  *Techniques avancées :*
  
  - *HyDE* : Hypothetical Document Embeddings
  - *Re-ranking* : Réordonnancement des résultats
  - *Query expansion* : Enrichissement de la requête
  - *Multi-query* : Plusieurs formulations
  - *Fusion* : Combinaison de plusieurs stratégies
]

#slide[
  == RAG vs Fine-tuning
  
  #table(
    columns: (1fr, 1fr, 1fr),
    [*Critère*], [*RAG*], [*Fine-tuning*],
    [Coût], [Faible], [Élevé],
    [Mise à jour], [Facile], [Difficile],
    [Données], [Documents], [Pairs Q/R],
    [Traçabilité], [Oui], [Non],
    [Latence], [Moyenne], [Faible],
  )
]

#slide[
  == Cas d'usage
  
  *Où utiliser le RAG ?*
  
  - Documentation technique (chatbot d'aide)
  - Support client (base de connaissances)
  - Recherche scientifique (analyse d'articles)
  - Juridique (recherche dans les lois)
  - Médical (informations à jour)
  - Entreprise (données internes)
]

#slide[
  == Frameworks et outils
  
  *Bibliothèques Python populaires :*
  
  - LangChain : Framework complet
  - LlamaIndex : Indexation et récupération
  - Haystack : Pipelines NLP
  - Transformers : Modèles HuggingFace
  - Sentence-Transformers : Embeddings
]

#slide[
  == Pipeline RAG simple
  
  ```python
  # 1. Charger documents
  documents = load_documents("data/")
  
  # 2. Créer embeddings
  embeddings = create_embeddings(documents)
  
  # 3. Stocker dans DB vectorielle
  vector_db = VectorStore(embeddings)
  
  # 4. Requête
  query = "Ma question"
  docs = vector_db.search(query, k=3)
  
  # 5. Générer réponse
  response = llm.generate(docs, query)
  ```
]

#slide[
  == Évaluation d'un système RAG
  
  *Métriques importantes :*
  
  - *Précision* : Documents pertinents récupérés
  - *Rappel* : Tous les documents pertinents trouvés
  - *MRR* : Mean Reciprocal Rank
  - *NDCG* : Normalized Discounted Cumulative Gain
  
  *Qualité de génération :*
  - Fidélité aux sources
  - Pertinence de la réponse
  - Cohérence
]

#slide[
  == Bonnes pratiques
  
  1. Choisir la bonne taille de chunks
  2. Utiliser des embeddings adaptés à la langue
  3. Nettoyer et prétraiter les documents
  4. Implémenter un système de cache
  5. Logger les requêtes pour amélioration
  6. Tester avec différents paramètres k
  7. Valider les sources retournées
]

#slide[
  == Considérations de production
  
  - *Scalabilité* : Gérer des millions de documents
  - *Performance* : Optimiser les temps de réponse
  - *Sécurité* : Contrôle d'accès aux documents
  - *Monitoring* : Suivre la qualité des réponses
  - *Coûts* : API calls, stockage, compute
]

#slide[
  == L'avenir du RAG
  
  *Tendances émergentes :*
  
  - RAG multi-modal (texte + images)
  - RAG conversationnel (mémorisation)
  - Agents autonomes avec RAG
  - RAG graphique (Knowledge Graphs)
  - Optimisation automatique des hyperparamètres
]

#slide[
  == Démonstration pratique
  
  *Nous allons voir :*
  
  1. Chargement de documents markdown
  2. Génération d'embeddings
  3. Recherche par similarité
  4. Génération de réponse avec un LLM
  
  #v(1em)
  
  Code disponible dans `rag_demo.py`
]

#slide[
  == Ressources pour aller plus loin
  
  *Documentation et tutoriels :*
  - LangChain documentation
  - HuggingFace courses
  - Papers : "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
  
  *Communautés :*
  - Discord LangChain
  - Forums HuggingFace
  - r/LocalLLaMA
]

#slide[
  == Questions ?
  
  #v(4em)
  
  #align(center)[
    *Merci pour votre attention !*
    
    #v(2em)
    
    Prêts pour la démonstration pratique ?
  ]
]