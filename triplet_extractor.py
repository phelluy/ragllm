"""
Module pour l'extraction et le parsing des triplets de connaissances.
"""

import logging
import re
from typing import List, Tuple
from config import TRIPLET_EXTRACT_PROMPT

logger = logging.getLogger(__name__)


class TripletExtractor:
    """
    Extrait les triplets (sujet, prédicat, objet) du texte via LLM.
    """

    def __init__(self, llm):
        """
        Initialise l'extracteur.
        
        Args:
            llm: Instance du LLM (OpenAILike ou compatible)
        """
        self.llm = llm

    def extract_raw_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Utilise le LLM pour extraire les triplets bruts du texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Liste de tuples (sujet, prédicat, objet)
        """
        try:
            prompt = TRIPLET_EXTRACT_PROMPT.format(text=text)
            response = self.llm.complete(prompt)
            response_text = response.text
            
            triplets = self._parse_triplets(response_text)
            
            if not triplets:
                logger.debug(f"Aucun triplet extrait, response brute: {response_text[:100]}...")
            
            return triplets
        except Exception as e:
            logger.error(f"Erreur extraction LLM: {e}")
            return []

    def _parse_triplets(self, response_text: str) -> List[Tuple[str, str, str]]:
        """
        Parse les triplets à partir de la réponse du LLM.
        
        Essaie plusieurs formats :
        1. Format regex: (sujet, prédicat, objet)
        2. Format flèche: sujet -> prédicat -> objet
        3. Format simplifié: ligne par ligne
        
        Args:
            response_text: Texte brut de la réponse
            
        Returns:
            Liste de tuples (sujet, prédicat, objet)
        """
        triplets = []
        
        # Tentative 1 : Regex pour (sujet, prédicat, objet)
        pattern_parentheses = r"\((.*?),(.*?),(.*?)\)"
        matches = re.findall(pattern_parentheses, response_text)
        for match in matches:
            subj, pred, obj = (m.strip() for m in match)
            if subj and pred and obj:
                triplets.append((subj, pred, obj))
        
        if triplets:
            logger.debug(f"Triplets extraits via regex parenthèses: {len(triplets)}")
            return triplets

        # Tentative 2 : Format avec flèches (sujet -> prédicat -> objet)
        lines = response_text.strip().split('\n')
        for line in lines:
            if '->' in line:
                parts = [p.strip() for p in line.split('->')]
                if len(parts) == 3 and all(parts):
                    triplets.append(tuple(parts))
        
        if triplets:
            logger.debug(f"Triplets extraits via flèches: {len(triplets)}")
            return triplets

        # Tentative 3 : Format simplifié (chaque ligne est un triplet séparé)
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Essayer de détecter un triplet simple
            # Format: "sujet relation objet" ou "sujet, relation, objet"
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3 and all(p for p in parts[:3]):
                    triplets.append(tuple(parts[:3]))

        return triplets

    @staticmethod
    def validate_triplet(subj: str, pred: str, obj: str) -> bool:
        """
        Valide un triplet (évite les auto-références et les triplets vides).
        
        Args:
            subj: Sujet
            pred: Prédicat
            obj: Objet
            
        Returns:
            True si le triplet est valide, False sinon
        """
        return bool(subj and pred and obj and subj.lower().strip() != obj.lower().strip())
