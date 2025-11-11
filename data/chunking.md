# Stratégies de Chunking

Le chunking consiste à découper des documents en morceaux de taille appropriée pour l'indexation et la récupération.

## Pourquoi le chunking ?

Les LLM ont une limite de contexte (nombre de tokens). Il faut donc :
- Découper les longs documents
- Créer des unités d'information cohérentes
- Optimiser la pertinence des résultats de recherche

## Stratégies de découpage

### Par taille fixe
Découpe le texte en morceaux de n caractères ou tokens.

**Avantages :**
- Simple à implémenter
- Prévisible

**Inconvénients :**
- Peut couper au milieu d'une phrase
- Perte de contexte

### Par paragraphe
Utilise la structure naturelle du document.

**Avantages :**
- Respect de la structure logique
- Chunks cohérents

**Inconvénients :**
- Taille variable
- Certains paragraphes peuvent être trop longs

### Par section/chapitre
Utilise les titres et sous-titres pour découper.

**Avantages :**
- Unités thématiques cohérentes
- Facilite la traçabilité

**Inconvénients :**
- Certaines sections peuvent être très longues
- Nécessite une structure claire du document

### Avec chevauchement (overlap)
Ajoute un recouvrement entre chunks consécutifs.

**Avantages :**
- Préserve le contexte aux frontières
- Améliore la récupération

**Inconvénients :**
- Duplication d'information
- Augmente la taille de la base

## Paramètres importants

### Taille du chunk
- Trop petit (< 100 tokens) : perte de contexte
- Trop grand (> 1000 tokens) : dilution de l'information
- Recommandation : 200-500 tokens

### Taille de l'overlap
- Généralement 10-20% de la taille du chunk
- Exemple : chunk de 500 tokens, overlap de 50-100 tokens

## Chunking sémantique

Techniques avancées qui utilisent le sens pour découper :
- Détection de changements de sujet
- Segmentation par embeddings
- Analyse de cohérence

## Bonnes pratiques

1. Tester plusieurs stratégies sur vos données
2. Adapter la taille aux types de questions
3. Maintenir les métadonnées (source, position, etc.)
4. Préserver les informations structurelles importantes
5. Évaluer l'impact sur la qualité des réponses
