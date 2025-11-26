#!/usr/bin/env python3
"""
Démonstration simple d'un système RAG (Retrieval-Augmented Generation)

Ce script illustre les concepts fondamentaux du RAG :
1. Chargement et indexation de documents
2. Création d'embeddings
3. Recherche par similarité
4. Génération de réponse avec un LLM via API REST compatible OpenAI
"""

import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import requests
import json
import urllib3

from llm_providers import get_provider, PROVIDERS

# Désactiver les avertissements SSL pour les connexions localhost
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class SimpleRAG:
    """Système RAG minimaliste pour la démonstration, avec sélection modulaire de provider LLM."""

    def __init__(
        self,
        data_dir: str = "data",
        provider_name: Optional[str] = None,
        override_model: Optional[str] = None,
        override_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialise le système RAG.

        Args:
            data_dir: Répertoire contenant les documents markdown.
            provider_name: Nom du provider (clé du dictionnaire PROVIDERS). Si None => provider par défaut.
            override_model: Permet de forcer un nom de modèle différent.
            override_url: Permet de remplacer l'URL de l'endpoint.
            api_key: Clé API explicite (sinon variable d'environnement définie dans le provider).
        """
        self.data_dir = data_dir
        self.documents = []
        self.embeddings = []

        # Modèle d'embedding multilingue pour meilleure performance en français
        print("Chargement du modèle d'embedding...")
        self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Configuration du provider LLM
        self._provider_cfg = get_provider(
            name=provider_name,
            override_model=override_model,
            override_url=override_url,
            api_key=api_key,
        )
        self.api_url = self._provider_cfg.url
        self.model = self._provider_cfg.model
        self.api_key = self._provider_cfg.api_key_env and os.getenv(self._provider_cfg.api_key_env, "") or ""
        self.use_api = True  # Toujours via API pour la génération

        print(f"Provider sélectionné: {provider_name or self._provider_cfg.name} | URL: {self.api_url} | Modèle: {self.model}")
        if self._provider_cfg.api_key_env:
            if not self.api_key:
                print(
                    f"⚠️ Clé API manquante pour {self._provider_cfg.name}. Définissez la variable d'environnement '{self._provider_cfg.api_key_env}'."
                )
            else:
                print(f"Clé API chargée depuis '{self._provider_cfg.api_key_env}'.")
        
    def load_documents(self) -> None:
        """Charge tous les fichiers markdown du répertoire data"""
        print(f"\nChargement des documents depuis {self.data_dir}/...")
        
        md_files = glob.glob(os.path.join(self.data_dir, "*.md"))
        
        for filepath in md_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Découpage simple par paragraphe
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                for para in paragraphs:
                    # Garder tous les paragraphes, même courts
                    self.documents.append({
                        'text': para,
                        'source': os.path.basename(filepath)
                    })
        
        print(f"  → {len(self.documents)} chunks chargés depuis {len(md_files)} fichiers")
    
    def create_embeddings(self) -> None:
        """Génère les embeddings pour tous les documents"""
        print("\nCréation des embeddings...")
        
        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"  → {len(self.embeddings)} embeddings créés (dimension: {self.embeddings.shape[1]})")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        Recherche les documents les plus similaires à la requête
        
        Args:
            query: Question de l'utilisateur
            top_k: Nombre de documents à retourner
            
        Returns:
            Liste de tuples (document, score de similarité)
        """
        # Générer l'embedding de la requête
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Calculer la similarité cosinus avec tous les documents
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Obtenir les indices des top-k documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Retourner les documents avec leurs scores
        results = [
            (self.documents[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def generate_response(self, query: str, context_docs: List[dict]) -> str:
        """
        Génère une réponse basée sur les documents récupérés (mode sans LLM)
        
        Méthode de secours qui retourne simplement le contexte formaté.
        En mode normal, utilisez generate_with_llm() qui appelle l'API REST.
        
        Args:
            query: Question de l'utilisateur
            context_docs: Documents récupérés
            
        Returns:            Réponse générée
        """
        # Construction du contexte
        context = "\n\n".join([
            f"[Source: {doc['source']}]\n{doc['text']}"
            for doc in context_docs
        ])
        
        # Retourne simplement le contexte formaté
        response = f"""Basé sur les documents suivants, voici des informations pertinentes :

{context}

---
Note : Cette réponse est basée sur les {len(context_docs)} documents les plus pertinents trouvés.
Pour une réponse générée par LLM, assurez-vous que l'API REST est accessible.
"""
        return response
    
    def configure_provider(
        self,
        provider_name: str,
        override_model: Optional[str] = None,
        override_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Change dynamiquement la configuration du provider LLM.

        Args:
            provider_name: Nom du provider dans PROVIDERS.
            override_model: Surcharge du modèle.
            override_url: Surcharge de l'URL.
            api_key: Clé API explicite (prioritaire sur variable d'env).
        """
        cfg = get_provider(
            name=provider_name,
            override_model=override_model,
            override_url=override_url,
            api_key=api_key,
        )
        self._provider_cfg = cfg
        self.api_url = cfg.url
        self.model = override_model or cfg.model
        self.api_key = api_key if api_key is not None else (cfg.api_key_env and os.getenv(cfg.api_key_env, "") or "")
        print(f"Provider reconfiguré: {provider_name} | URL: {self.api_url} | Modèle: {self.model}")
        if cfg.api_key_env:
            if not self.api_key:
                print(
                    f"⚠️ Clé API manquante pour {cfg.name}. Définissez la variable d'environnement '{cfg.api_key_env}'."
                )
            else:
                print(f"Clé API chargée depuis '{cfg.api_key_env}'.")
    
    def generate_with_llm(self, query: str, context_docs: List[dict]) -> str:
        """
        Génère une réponse en utilisant l'API REST
        
        Args:
            query: Question de l'utilisateur
            context_docs: Documents récupérés
            
        Returns:
            Réponse générée par l'API
        """
        # Construction du contexte
        context = "\n\n".join([doc['text'] for doc in context_docs])
        
        # Construction du prompt
        prompt = f"""Basé sur le contexte suivant, réponds à la question de manière concise et précise.

Contexte:
{context}

Question: {query}

Réponse:"""
        
        # Ajouter l'instruction pour désactiver la réflexion si nécessaire
        #prompt = prompt + " /no_think"
        
        # Préparation de la requête pour l'API OpenAI-compatible
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9,
        }
        
        try:
            # Appel à l'API REST (sans vérification SSL pour localhost)
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            verify_ssl = self.api_url.startswith("https://")
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                verify=verify_ssl,
                timeout=60,
            )
            response.raise_for_status()
            #print("response:", response.json())
            
            # Extraction de la réponse
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                return "Erreur : format de réponse inattendu de l'API"
                
        except requests.exceptions.RequestException as e:
            return f"Erreur lors de l'appel à l'API : {str(e)}"
    
    def interactive_demo(self) -> None:
        """Démo interactive permettant de poser des questions"""
        print("\n" + "="*70)
        print("SYSTÈME RAG - MODE INTERACTIF")
        print("="*70)
        print("\nPosez vos questions sur le RAG et les LLM.")
        print("Tapez 'quit' ou 'exit' pour quitter.\n")
        
        while True:
            query = input("❓ Votre question : ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nAu revoir !")
                break
            
            if not query:
                continue
            
            # Recherche des documents pertinents
            print("\n🔍 Recherche des documents pertinents...")
            results = self.search(query, top_k=20)
            
            print(f"\n📚 Documents trouvés (top 3):")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n  [{i}] Score: {score:.4f} | Source: {doc['source']}")
                print(f"      {doc['text'][:150]}...")
            
            # Génération de la réponse
            print("\n💬 Réponse générée:")
            print("-" * 70)
            
            context_docs = [doc for doc, score in results]
            
            # Toujours utiliser l'API pour la génération
            response = self.generate_with_llm(query, context_docs)
            
            print(response)
            print("-" * 70)
            print()


def main():
    """Fonction principale de démonstration."""
    print("=" * 70)
    print("DÉMONSTRATION D'UN SYSTÈME RAG SIMPLE")
    print("=" * 70)

    # Sélection optionnelle du provider via variable d'environnement LLM_PROVIDER
    provider_env = os.getenv("LLM_PROVIDER")  # ex: MISTRAL_LARGE
    if provider_env and provider_env not in PROVIDERS:
        print(f"⚠️ Provider '{provider_env}' inconnu. Providers disponibles: {list(PROVIDERS.keys())}")
        provider_env = None

    # Initialisation du système RAG avec provider modulaire
    rag = SimpleRAG(data_dir="data_big", provider_name=provider_env)
    
    # Chargement et indexation des documents
    rag.load_documents()
    rag.create_embeddings()
    
    # Pour changer dynamiquement le provider pendant la session :
    # rag.configure_provider("MISTRAL_LARGE")
    # rag.configure_provider("PALGANIA_QWEN3", override_model="Qwen3-72B")
    
    # Exemples de requêtes
    print("\n" + "="*70)
    print("EXEMPLES DE REQUÊTES")
    print("="*70)
    
    example_queries = [
        "Qu'est-ce que le RAG ?",
        "Comment fonctionnent les embeddings ?",
        "Quelles sont les bases de données vectorielles ?",
    ]
    
    for query in example_queries:
        print(f"\n❓ Question : {query}")
        print("-" * 70)
        
        results = rag.search(query, top_k=3)
        
        print(f"📚 Top 3 documents pertinents :\n")
        for i, (doc, score) in enumerate(results, 1):
            print(f"[{i}] Score: {score:.4f} | Source: {doc['source']}")
            print(f"    {doc['text'][:200]}...")
            print()
    
    # Mode interactif
    print("\n" + "="*70)
    response = input("\nVoulez-vous essayer le mode interactif ? (o/n) : ").strip().lower()
    if response in ['o', 'y', 'oui', 'yes']:
        rag.interactive_demo()
    else:
        print("\nDémonstration terminée. Au revoir !")


if __name__ == "__main__":
    main()
