# 📋 Refactorisation du Code RAG Graph

## 🎯 Vue d'ensemble

Le code principal `rag_graph.py` a été **restructuré et amélioré** pour une meilleure maintenabilité, clarté et extensibilité. Les changements incluent :

### ✨ Principaux changements

1. **Modularisation** : Séparation des responsabilités en modules spécialisés
2. **Configuration centralisée** : Toutes les constantes dans `config.py`
3. **Gestion des erreurs améliorée** : Logging structuré via Python's `logging` module
4. **Extraction des triplets robuste** : Parsing multi-format avec fallbacks
5. **Gestion Neo4j dédiée** : Classe `Neo4jManager` avec context manager
6. **Extraction des prompts flexible** : Module `PromptExtractor` pour les différents formats
7. **Documentation complète** : Docstrings détaillées pour chaque classe et méthode

---

## 📂 Structure des fichiers

### Nouveaux fichiers créés

```
ragllm/
├── config.py                  # Configuration centralisée
├── entity_normalizer.py       # Normalisation des entités
├── triplet_extractor.py       # Extraction des triplets
├── prompt_extractor.py        # Extraction des prompts
├── neo4j_manager.py           # Gestion Neo4j
└── rag_graph.py              # Code principal refactorisé
```

---

## 🔧 Configuration (`config.py`)

Tous les paramètres sont maintenant centralisés :

```python
# Modèles
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Seuils
ENTITY_SIMILARITY_THRESHOLD = 0.85
MAX_TRIPLETS_PER_CHUNK = 15

# Neo4j
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Prompts
TRIPLET_EXTRACT_PROMPT = "..."
GRAPH_QA_PROMPT = "..."
```

**Avantage** : Modifiez les paramètres sans toucher au code principal !

---

## 🔤 Normalisation des entités (`entity_normalizer.py`)

Classe complètement extraite et améliorée :

```python
from entity_normalizer import EntityNormalizer

normalizer = EntityNormalizer()
canonical = normalizer.normalize("Jules Amaro")  # → "Jules Amaro" (canonical)

# Obtenir les statistiques
stats = normalizer.get_statistics()
print(f"Entités canoniques: {stats['canonical_entities']}")
print(f"Mentions totales: {stats['total_mentions']}")
```

**Améliorations** :
- Gestion d'erreurs robuste avec logging
- Méthode `get_statistics()` pour analyser les résultats
- Configuration du modèle centralisée

---

## 📊 Extraction des triplets (`triplet_extractor.py`)

Parsing multi-format avec fallbacks intelligents :

```python
from triplet_extractor import TripletExtractor

extractor = TripletExtractor(llm)
triplets = extractor.extract_raw_triplets(text)

# Validation robuste
is_valid = TripletExtractor.validate_triplet(subject, predicate, obj)
```

**Formats supportés** :
1. `(sujet, prédicat, objet)` (regex)
2. `sujet -> prédicat -> objet` (flèches)
3. `sujet, prédicat, objet` (virgules)

---

## 🔐 Gestion Neo4j (`neo4j_manager.py`)

Interface propre pour Neo4j :

```python
from neo4j_manager import Neo4jManager

# Utilisation simple
with Neo4jManager() as manager:
    manager.connect()
    count = manager.count_nodes()
    manager.clear_database()

# Ou manuellement
manager = Neo4jManager()
if manager.connect():
    manager.clear_database()
manager.close()
```

**Fonctionnalités** :
- Connexion sécurisée avec timeout
- Méthodes pratiques (`count_nodes()`, `graph_exists()`)
- Context manager pour gestion automatique
- Logging structuré des erreurs

---

## 💬 Extraction des prompts (`prompt_extractor.py`)

Récupère les prompts depuis les événements LlamaIndex :

```python
from prompt_extractor import PromptExtractor

extractor = PromptExtractor()

# Depuis un payload
text = extractor.extract_from_payload(event.payload)

# Depuis plusieurs événements
full_text = extractor.extract_from_events(events)
```

**Supporte** :
- `formatted_prompt` direct
- Messages structurés
- Blocks imbriqués
- Fallback gracieux

---

## 🚀 Utilisation principale (`rag_graph.py`)

### Initialisation

```python
from rag_graph import GraphRAGDemo

demo = GraphRAGDemo(
    data_dir="data",
    provider_name="MISTRAL_NEMO",
    use_neo4j=False,
    top_k=7,
    reload=False
)
```

### Indexation

```python
demo.load_and_index()  # Charge ou construit les index
```

### Requête

```python
demo.query("Pourquoi se méfier de Jules ?")
```

### Mode interactif

```python
demo.interactive_loop()  # Boucle interactive
```

---

## 📊 Améliorations techniques

### Logging structuré

**Avant** :
```python
print("Erreur!")
print(f"Result: {result}")
```

**Après** :
```python
logger.error("Erreur détaillée")
logger.info(f"Résultat: {result}")
logger.debug(f"Debug: {details}")
```

### Gestion des erreurs robuste

Chaque module gère ses erreurs proprement :

```python
def extract_raw_triplets(self, text: str):
    try:
        # Logique
    except Exception as e:
        logger.error(f"Erreur extraction: {e}")
        return []  # Fallback sûr
```

### Type hints complets

```python
def extract_from_payload(payload) -> str:
    """Extract prompt from payload."""
```

---

## ⚙️ Configuration en ligne de commande

```bash
# Utilisation basique
python rag_graph.py

# Avec provider spécifique
python rag_graph.py --provider MISTRAL_NEMO

# Avec Neo4j
python rag_graph.py --neo4j

# Forcer la reconstruction
python rag_graph.py --reload

# Nombre de chunks personnalisé
python rag_graph.py --top-k 10

# Sans mode interactif
python rag_graph.py --no-interactive
```

---

## 📈 Performance et optimisations

1. **Caching des embeddings** : Les normalisateurs réutilisent les embeddings
2. **Parsing intelligent** : Plusieurs formats supportés pour robustesse
3. **Logging modulé** : Désactivez le debug pour plus de performance
4. **Context managers** : Gestion automatique des ressources

---

## 🐛 Debugging

### Activer les logs complets

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspecter les index

```python
# Statistiques d'entités
stats = demo.entity_normalizer.get_statistics()
print(stats)

# Nombre de nœuds
if demo.use_neo4j:
    count = demo.neo4j_manager.count_nodes()
    print(f"Nœuds: {count}")
```

---

## 🔄 Migration depuis l'ancienne version

Si vous aviez du code utilisant l'ancienne version :

**Avant** :
```python
normalizer = EntityNormalizer()  # Importé du main
text = normalizer.normalize("Jules")
```

**Après** :
```python
from entity_normalizer import EntityNormalizer
normalizer = EntityNormalizer()
text = normalizer.normalize("Jules")
```

---

## ✅ Checklist de validation

- [x] Pas d'erreurs de syntaxe
- [x] Tous les modules importent correctement
- [x] Logging fonctionne partout
- [x] Type hints complets
- [x] Docstrings détaillées
- [x] Gestion des erreurs robuste
- [x] Configuration centralisée
- [x] Code main simplifié

---

## 📝 Notes finales

Cette refactorisation rend le code :
- ✅ **Plus lisible** : Responsabilités claires
- ✅ **Plus testable** : Modules indépendants
- ✅ **Plus maintenable** : Configuration centralisée
- ✅ **Plus robuste** : Gestion d'erreurs améliorée
- ✅ **Plus extensible** : Facile d'ajouter nouvelles fonctionnalités

Enjoy ! 🚀
