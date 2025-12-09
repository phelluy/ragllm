"""
Module pour l'extraction et la gestion des prompts depuis les événements de callback.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class PromptExtractor:
    """
    Extrait les prompts depuis les payloads d'événements LlamaIndex.
    """

    @staticmethod
    def extract_from_payload(payload) -> str:
        """
        Extrait le contenu textuel d'un payload EventPayload.
        
        Essaie plusieurs sources :
        1. formatted_prompt
        2. messages (avec reconstruction)
        3. fallback sur str(payload)
        
        Args:
            payload: EventPayload de LlamaIndex
            
        Returns:
            Texte du prompt ou message d'erreur
        """
        try:
            # Tentative 1 : formatted_prompt
            if isinstance(payload, dict) and "formatted_prompt" in payload:
                return payload["formatted_prompt"]
            
            if hasattr(payload, "formatted_prompt"):
                return payload.formatted_prompt

            # Tentative 2 : messages
            if isinstance(payload, dict) and "messages" in payload:
                messages = payload["messages"]
                text = PromptExtractor._reconstruct_from_messages(messages)
                if text:
                    return text
            
            if hasattr(payload, "messages"):
                messages = payload.messages
                text = PromptExtractor._reconstruct_from_messages(messages)
                if text:
                    return text

            # Fallback
            logger.info(f"DEBUG: Payload type: {type(payload)}")
            logger.info(f"DEBUG: Payload attrs: {dir(payload)}")
            if isinstance(payload, dict):
                 logger.info(f"DEBUG: Payload keys: {payload.keys()}")
            
            return str(payload)
        except Exception as e:
            logger.error(f"Erreur extraction prompt: {e}")
            return str(payload)

    @staticmethod
    def _reconstruct_from_messages(messages) -> str:
        """
        Reconstruit un texte lisible à partir d'une liste de messages.
        
        Args:
            messages: Liste de messages
            
        Returns:
            Texte formaté ou chaîne vide
        """
        if not isinstance(messages, list):
            return ""
        
        text_parts = []
        for msg in messages:
            try:
                # Récupérer le rôle
                role = getattr(msg, 'role', 'unknown')
                
                # Récupérer le contenu
                content = None
                if hasattr(msg, 'content'):
                    content = msg.content
                elif hasattr(msg, 'blocks'):
                    # Pour les messages structurés avec blocks
                    block_texts = []
                    for block in msg.blocks:
                        if hasattr(block, 'text'):
                            block_texts.append(block.text)
                    content = " ".join(block_texts) if block_texts else None
                
                if content:
                    text_parts.append(f"[{role}]: {content}")
            except Exception as e:
                logger.debug(f"Erreur reconstruction message: {e}")
                continue
        
        return "\n\n".join(text_parts) if text_parts else ""

    @staticmethod
    def extract_from_events(events: List) -> str:
        """
        Extrait et compile les prompts de plusieurs événements.
        
        Args:
            events: Liste d'événements LlamaIndex
            
        Returns:
            Texte compilé des prompts
        """
        prompts = []
        for i, event in enumerate(events):
            try:
                # Les événements peuvent avoir une structure (event, payload)
                if isinstance(event, tuple) and len(event) > 0:
                    event_obj = event[0]
                else:
                    event_obj = event
                
                # Extraire le payload
                payload = getattr(event_obj, 'payload', None)
                if payload:
                    text = PromptExtractor.extract_from_payload(payload)
                    prompts.append(f"--- Event {i+1} ---\n{text}\n")
            except Exception as e:
                logger.debug(f"Erreur traitement événement: {e}")
                continue
        
        return "\n".join(prompts) if prompts else "Pas de prompt trouvé"
