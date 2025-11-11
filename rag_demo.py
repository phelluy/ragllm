#!/usr/bin/env python3
"""
Démonstration simple d'un système RAG (Retrieval-Augmented Generation)

Ce script illustre les concepts fondamentaux du RAG :
1. Chargement et indexation de documents
2. Création d'embeddings
3. Recherche par similarité
4. Génération de réponse avec un LLM
"""

import os
import glob
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class SimpleRAG:
    """Système RAG minimaliste pour la démonstration"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialise le système RAG
        
        Args:
            data_dir: Répertoire contenant les documents markdown
        """
        self.data_dir = data_dir
        self.documents = []
        self.embeddings = []
        
        # Modèle d'embedding léger et performant
        print("Chargement du modèle d'embedding...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # LLM léger pour la génération (optionnel, commenté par défaut)
        # Décommenter si vous avez assez de mémoire
        self.llm_tokenizer = None
        self.llm_model = None
        
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
                    # Filtrer les paragraphes trop courts (titres, etc.)
                    if len(para) > 50:
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
        Génère une réponse basée sur les documents récupérés
        
        Pour cette démo, on utilise une génération simple sans LLM.
        En production, on utiliserait un vrai LLM.
        
        Args:
            query: Question de l'utilisateur
            context_docs: Documents récupérés
            
        Returns:
            Réponse générée
        """
        # Construction du contexte
        context = "\n\n".join([
            f"[Source: {doc['source']}]\n{doc['text']}"
            for doc in context_docs
        ])
        
        # Pour cette démo, on retourne simplement le contexte formaté
        # En production, on passerait ceci à un LLM
        response = f"""Basé sur les documents suivants, voici des informations pertinentes :

{context}

---
Note : Cette réponse est basée sur les {len(context_docs)} documents les plus pertinents trouvés.
Pour une réponse générée par LLM, décommentez la section d'initialisation du modèle de génération.
"""
        return response
    
    def load_llm(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> None:
        """
        Charge un LLM pour la génération (optionnel)
        
        Args:
            model_name: Nom du modèle HuggingFace à charger
        """
        print(f"\nChargement du LLM {model_name}...")
        print("  (Ceci peut prendre quelques minutes la première fois)")
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        print("  → LLM chargé avec succès")
    
    def generate_with_llm(self, query: str, context_docs: List[dict]) -> str:
        """
        Génère une réponse en utilisant un LLM
        
        Args:
            query: Question de l'utilisateur
            context_docs: Documents récupérés
            
        Returns:
            Réponse générée par le LLM
        """
        if self.llm_model is None:
            return self.generate_response(query, context_docs)
        
        # Construction du prompt
        context = "\n\n".join([doc['text'] for doc in context_docs])
        
        prompt = f"""Basé sur le contexte suivant, réponds à la question de manière concise et précise.

Contexte:
{context}

Question: {query}

Réponse:"""
        
        # Tokenisation
        inputs = self.llm_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Génération
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )
        
        # Décodage
        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraction de la réponse (après "Réponse:")
        if "Réponse:" in response:
            response = response.split("Réponse:")[-1].strip()
        
        return response
    
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
            results = self.search(query, top_k=3)
            
            print(f"\n📚 Documents trouvés (top 3):")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n  [{i}] Score: {score:.4f} | Source: {doc['source']}")
                print(f"      {doc['text'][:150]}...")
            
            # Génération de la réponse
            print("\n💬 Réponse générée:")
            print("-" * 70)
            
            context_docs = [doc for doc, score in results]
            
            if self.llm_model is not None:
                response = self.generate_with_llm(query, context_docs)
            else:
                response = self.generate_response(query, context_docs)
            
            print(response)
            print("-" * 70)
            print()


def main():
    """Fonction principale de démonstration"""
    print("="*70)
    print("DÉMONSTRATION D'UN SYSTÈME RAG SIMPLE")
    print("="*70)
    
    # Initialisation du système RAG
    rag = SimpleRAG(data_dir="data")
    
    # Chargement et indexation des documents
    rag.load_documents()
    rag.create_embeddings()
    
    # Optionnel : charger un LLM pour la génération
    # Décommentez la ligne suivante si vous avez assez de ressources
    # rag.load_llm()
    
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
