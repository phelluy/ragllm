"""
Module pour la normalisation et fusion des entités similaires.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import ENTITY_NORMALIZER_MODEL, ENTITY_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """
    Normalise et fusionne les entités similaires.
    
    Utilise un modèle de sentence-transformers pour encoder les entités et
    détecte les variations/synonymes en utilisant une similarité sémantique.
    """

    def __init__(self, model_name: str = ENTITY_NORMALIZER_MODEL, threshold: float = ENTITY_SIMILARITY_THRESHOLD):
        """
        Initialise le normaliseur d'entités.
        
        Args:
            model_name: Nom du modèle sentence-transformers à utiliser
            threshold: Seuil de similarité (0.0-1.0) pour fusionner deux entités
        """
        logger.info(f"🧠 Chargement du modèle de normalisation d'entités ({model_name})...")
        self.embed_model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.entity_index = {}      # {canonical_name: [aliases]}
        self.entity_embeddings = {} # {canonical_name: embedding}

    def normalize(self, entity: str) -> str:
        """
        Trouve l'entité canonique pour une mention donnée.
        
        Processus :
        1. Nettoyage basique (strip, case-insensitive)
        2. Vérification contre les alias connus (exact match)
        3. Recherche par similarité sémantique
        4. Création d'une nouvelle entité canonique si nécessaire
        
        Args:
            entity: L'entité à normaliser
            
        Returns:
            L'entité canonique (existante ou nouvelle)
        """
        # Nettoyage basique
        entity_clean = entity.strip()
        entity_lower = entity_clean.lower()

        # 1. Vérifier si c'est un alias connu (exact match, case insensitive)
        for canonical, aliases in self.entity_index.items():
            if entity_lower in [a.lower() for a in aliases]:
                return canonical
        
        # 2. Si pas trouvé, calculer l'embedding
        try:
            emb = self.embed_model.encode([entity_clean])[0]
        except Exception as e:
            logger.warning(f"Erreur lors du calcul de l'embedding pour '{entity_clean}': {e}")
            # Fallback : créer une nouvelle entité
            if entity_clean not in self.entity_index:
                self.entity_index[entity_clean] = [entity_clean]
            return entity_clean

        # 3. Chercher une similarité sémantique forte avec les entités existantes
        best_match = None
        best_sim = 0.0
        
        if self.entity_embeddings:
            try:
                canonicals = list(self.entity_embeddings.keys())
                embeddings = np.array(list(self.entity_embeddings.values()))
                
                sims = cosine_similarity([emb], embeddings)[0]
                max_idx = np.argmax(sims)
                best_sim = float(sims[max_idx])
                
                if best_sim > self.threshold:
                    best_match = canonicals[max_idx]
            except Exception as e:
                logger.warning(f"Erreur lors de la similarité pour '{entity_clean}': {e}")

        if best_match:
            # Fusionner avec la plus similaire
            logger.debug(f"Normalisation : '{entity_clean}' -> '{best_match}' (score: {best_sim:.2f})")
            if entity_clean not in self.entity_index[best_match]:
                self.entity_index[best_match].append(entity_clean)
            return best_match
        else:
            # Créer une nouvelle entité canonique
            logger.debug(f"Nouvelle entité : '{entity_clean}'")
            self.entity_index[entity_clean] = [entity_clean]
            self.entity_embeddings[entity_clean] = emb
            return entity_clean

    def get_statistics(self) -> dict:
        """
        Retourne des statistiques sur les entités normalisées.
        
        Returns:
            Dictionnaire avec nombre d'entités canoniques et d'aliases
        """
        total_aliases = sum(len(aliases) for aliases in self.entity_index.values())
        return {
            "canonical_entities": len(self.entity_index),
            "total_mentions": total_aliases,
            "avg_aliases_per_entity": total_aliases / len(self.entity_index) if self.entity_index else 0
        }
