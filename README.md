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
- **Transformers** : Pour les modèles HuggingFace
- **PyTorch** : Backend pour les modèles
- **NumPy** : Calculs numériques
- **Typst** : Système de composition des slides

## Notes

- Le LLM de génération est optionnel (commenté par défaut) pour économiser la mémoire
- La première exécution téléchargera les modèles (~100 MB pour l'embedding)
- Les modèles sont mis en cache automatiquement par HuggingFace

## Licence

Voir le fichier LICENSE
