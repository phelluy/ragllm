# ragllm
A short introduction to RAG for LLM

## Description

Ce dépôt contient un cours d'introduction aux techniques de RAG (Retrieval-Augmented Generation) pour les LLM, destiné à un niveau M1.

## Structure du projet

```
ragllm/
├── doc/                          # Documentation et slides
│   └── slides_rag.typ           # Slides Typst (30 diapos)
├── data/                         # Documents markdown pour indexation
│   ├── introduction_rag.md      # Introduction au RAG
│   ├── embeddings.md            # Les embeddings
│   ├── bases_vectorielles.md    # Bases de données vectorielles
│   ├── chunking.md              # Stratégies de découpage
│   └── generation_llm.md        # Génération avec LLM
├── rag_demo.py                  # Démonstration RAG en Python
└── requirements.txt             # Dépendances Python
```

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/phelluy/ragllm.git
cd ragllm
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Compiler les slides

Pour compiler les slides Typst en PDF :
```bash
typst compile doc/slides_rag.typ
```

### Lancer la démonstration RAG

```bash
python rag_demo.py
```

Le script va :
1. Charger les documents markdown du dossier `data/`
2. Créer des embeddings avec le modèle `all-MiniLM-L6-v2`
3. Effectuer des recherches de similarité sur des exemples de questions
4. Proposer un mode interactif pour tester vos propres questions

### Mode interactif

Après les exemples, le script propose un mode interactif où vous pouvez poser vos propres questions sur le RAG.

## Contenu du cours

Les slides couvrent :
- Introduction et motivation
- Concepts fondamentaux du RAG
- Architecture d'un système RAG
- Embeddings et similarité
- Stockage vectoriel
- Récupération de documents
- Génération augmentée
- Mise en pratique

## Technologies utilisées

- **Sentence Transformers** : Pour les embeddings
- **PyTorch** : Backend pour les modèles d'embedding
- **NumPy** : Calculs numériques
- **Requests** : Pour les appels à l'API REST
- **Typst** : Système de composition des slides

## Configuration de l'API

Le système RAG utilise une API REST compatible OpenAI pour la génération de réponses.

- **URL par défaut** : `https://127.0.0.1:8080/v1/chat/completions`
- **Format** : Compatible avec l'API OpenAI Chat Completions
- **Authentification** : Aucune clé API requise (API locale)

Pour changer l'URL de l'API, vous pouvez :
```python
rag = SimpleRAG(api_url="https://autre-serveur:port/v1/chat/completions")
# ou après l'initialisation :
rag.configure_api("https://autre-serveur:port/v1/chat/completions")
```

## Notes

- La première exécution téléchargera le modèle d'embedding (~100 MB)
- Les modèles sont mis en cache automatiquement par HuggingFace
- L'API REST doit être accessible pour la génération de réponses
- Les embeddings sont calculés localement (pas via l'API)

## Licence

Voir le fichier LICENSE
