# Les Embeddings

Les embeddings sont des représentations vectorielles de texte qui capturent le sens sémantique des mots ou des phrases.

## Qu'est-ce qu'un embedding ?

Un embedding transforme du texte en un vecteur de nombres réels dans un espace multidimensionnel. Les textes ayant des significations similaires sont représentés par des vecteurs proches dans cet espace.

## Modèles d'embedding populaires

### Sentence-BERT (SBERT)
- Modèle basé sur BERT optimisé pour les embeddings de phrases
- Très performant pour la similarité sémantique
- Disponible en plusieurs tailles et langues

### all-MiniLM-L6-v2
- Modèle compact et rapide
- Dimension de vecteur : 384
- Bon compromis entre performance et vitesse

### multilingual-e5-base
- Support multilingue
- Performant sur de nombreuses langues
- Dimension de vecteur : 768

## Mesures de similarité

Pour comparer deux embeddings, on utilise généralement :

### Similarité cosinus
La mesure la plus courante. Elle calcule le cosinus de l'angle entre deux vecteurs. Une valeur proche de 1 indique une grande similarité.

### Distance euclidienne
Mesure la distance géométrique directe entre deux vecteurs dans l'espace.

### Produit scalaire
Multiplication élément par élément des vecteurs, puis somme des résultats.

## Utilisation pratique

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Mon texte à encoder")
```
