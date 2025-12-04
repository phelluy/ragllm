#!/usr/bin/env python3
"""
Démonstration avancée : RAG avec Graphe de Connaissances (Graph RAG) via LlamaIndex.

Ce script illustre :
1. L'utilisation de LlamaIndex pour l'ingestion et l'indexation.
2. La création d'un index vectoriel classique.
3. La création d'un index de graphe (Knowledge Graph).
4. La comparaison des résultats entre les deux approches.

Prérequis :
- pip install -r requirements.txt
- (Optionnel) Serveur Neo4j pour le stockage de graphe persistant.
"""

import os
import sys
import logging
import argparse
from typing import List, Optional
import networkx as nx

# Forcer le backend matplotlib non-interactif AVANT d'importer pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from neo4j import GraphDatabase

# Désactiver le parallelisme des tokenizers pour éviter les avertissements de fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration du logging pour voir ce qui se passe
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports LlamaIndex
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    QueryBundle,
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine

# Import du gestionnaire de providers existant
try:
    from llm_providers import get_provider, PROVIDERS
except ImportError:
    # Fallback si lancé depuis un autre dossier
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from llm_providers import get_provider, PROVIDERS

# Tentative d'import de Neo4j (optionnel)
try:
    from llama_index.graph_stores.neo4j import Neo4jGraphStore
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class HybridGraphRetriever(BaseRetriever):
    """Retriever hybride qui combine les triplets du graphe ET le texte source des chunks."""
    
    def __init__(self, graph_index, vector_index, similarity_top_k: int = 3):
        self.graph_index = graph_index
        self.vector_index = vector_index
        self.similarity_top_k = similarity_top_k
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Récupère à la fois les triplets du graphe ET le texte des chunks pertinents."""
        
        # 1. Récupérer les triplets du graphe
        graph_retriever = self.graph_index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            include_text=True,
        )
        graph_nodes = graph_retriever.retrieve(query_bundle)
        
        # 2. Récupérer les chunks de texte via l'index vectoriel
        vector_retriever = self.vector_index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        vector_nodes = vector_retriever.retrieve(query_bundle)
        
        # 3. Combiner les résultats
        # On garde les chunks vectoriels avec leur texte complet
        # et on ajoute les triplets du graphe comme contexte structuré
        all_nodes = []
        
        # Ajouter d'abord les chunks de texte (source primaire)
        all_nodes.extend(vector_nodes)
        
        # Ajouter ensuite les triplets du graphe comme contexte supplémentaire
        # On peut formater les triplets de manière plus lisible
        for node in graph_nodes:
            content = node.node.get_content()
            # Si c'est un triplet, on le formate pour être plus lisible
            if "->" in content or "(" in content:
                # C'est probablement un triplet, on le garde tel quel
                all_nodes.append(node)
        
        return all_nodes


class GraphRAGDemo:
    def __init__(
        self,
        data_dir: str = "data",
        provider_name: Optional[str] = None,
        use_neo4j: bool = False,
        neo4j_url: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        top_k: int = 3,
        reload: bool = False,
        persist_dir: str = "./storage",
    ):
        self.data_dir = data_dir
        self.top_k = top_k
        self.persist_dir = persist_dir
        self.force_rebuild = False  # Flag pour forcer la reconstruction
        
        # 1. Configuration du LLM via le provider existant
        provider_cfg = get_provider(name=provider_name)
        print(f"🔌 Provider LLM : {provider_cfg.name} ({provider_cfg.model})")
        
        # LlamaIndex utilise OpenAILike pour les endpoints compatibles
        # Note: api_base correspond à l'URL de base (sans /chat/completions)
        api_base = provider_cfg.url.replace("/chat/completions", "")
        api_key = provider_cfg.api_key() or "fake-key"
        
        self.llm = OpenAILike(
            model=provider_cfg.model,
            api_base=api_base,
            api_key=api_key,
            is_chat_model=True,
            context_window=4096, # Ajuster selon le modèle
            max_tokens=512,
        )
        
        # 2. Configuration de l'Embedding (Local pour correspondre à la démo 1)
        print("🧠 Chargement du modèle d'embedding (HuggingFace)...")
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Appliquer la configuration globale
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        
        # Configuration du Callback Manager pour le debug et la capture de prompt
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=False)
        callback_manager = CallbackManager([self.llama_debug])
        Settings.callback_manager = callback_manager
        
        # 3. Configuration du Graph Store
        self.use_neo4j = use_neo4j
        if use_neo4j:
            if not NEO4J_AVAILABLE:
                print("⚠️ Neo4j demandé mais librairie manquante ou erreur d'import. Fallback sur SimpleGraphStore.")
                self.graph_store = SimpleGraphStore()
            else:
                print(f"🕸️ Connexion à Neo4j ({neo4j_url})...")
                if not self.reload:
                    self.clear_neo4j_database(neo4j_url, neo4j_user, neo4j_password)
                try:
                    self.graph_store = Neo4jGraphStore(
                        username=neo4j_user,
                        password=neo4j_password,
                        url=neo4j_url,
                        database="neo4j",
                    )
                except Exception as e:
                    print(f"❌ Erreur connexion Neo4j: {e}. Fallback sur SimpleGraphStore.")
                    self.graph_store = SimpleGraphStore()
        else:
            print("💾 Utilisation du stockage de graphe en mémoire (SimpleGraphStore).")
            self.graph_store = SimpleGraphStore()
            
        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        
        self.vector_index = None
        self.graph_index = None

    def clear_neo4j_database(self, url, user, password):
        """Vide la base de données Neo4j pour éviter les doublons."""
        print("🧹 Nettoyage de la base Neo4j...")
        try:
            driver = GraphDatabase.driver(url, auth=(user, password))
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("   → Base vidée avec succès.")
            driver.close()
        except Exception as e:
            print(f"⚠️ Attention: Impossible de vider Neo4j: {e}")

    def load_and_index(self):
        """Charge les documents et construit les index (ou les recharge si déjà existants).
        
        Comportement :
        - Par défaut : Si index existe → charge. Sinon → construit.
        - Avec force_rebuild=True : Force la reconstruction complète.
        """
        
        # Déterminer si les index existent déjà
        if not self.force_rebuild:
            graph_exists = self._check_graph_exists()
            indexes_persisted = os.path.exists(self.persist_dir)
            
            # Si les index existent, les charger
            if graph_exists and indexes_persisted:
                print(f"\n✅ Index existants détectés. Chargement depuis le stockage...")
                self._load_existing_indexes()
                return
        else:
            print(f"\n🔄 Reconstruction forcée demandée.")
        
        # Sinon, reconstruire tout
        print(f"\n📂 Chargement des documents depuis {self.data_dir}...")
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        print(f"   → {len(documents)} documents chargés.")
        
        # Vider Neo4j avant reconstruction
        if self.use_neo4j and NEO4J_AVAILABLE:
            print("🧹 Nettoyage de Neo4j avant reconstruction...")
            self.clear_neo4j_database("bolt://localhost:7687", "neo4j", "password")
        
        # Construction de l'index Vectoriel
        print("\n🚀 Construction de l'index VECTORIEL...")
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=True
        )
        self.vector_index.set_index_id("vector_index")
        
        # Construction de l'index Graph
        print("\n🕸️ Construction de l'index GRAPHE (Extraction des triplets)...")
        print("   (Ceci peut prendre du temps car le LLM doit extraire les entités)")
        
        self.graph_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=10, 
            include_embeddings=True,
            show_progress=True,
        )
        self.graph_index.set_index_id("graph_index")
        
        print("✅ Indexation terminée.")
        
        # Persistance
        print(f"\n💾 Sauvegarde des index dans {self.persist_dir}...")
        self.storage_context.persist(persist_dir=self.persist_dir)
        print("   → Sauvegarde terminée.")
        
        # Générer l'image du graphe (pour Neo4j ET SimpleGraphStore)
        self.generate_graph_image()
    
    def _extract_prompt_from_payload(self, payload) -> str:
        """Extrait le contenu textuel d'un payload EventPayload de manière lisible."""
        try:
            # Essayer de récupérer formatted_prompt en premier
            if "formatted_prompt" in payload:
                return payload["formatted_prompt"]
            
            # Sinon, chercher les messages
            if "messages" in payload:
                messages = payload["messages"]
                if isinstance(messages, list):
                    text_parts = []
                    for msg in messages:
                        # Extraire le texte de chaque message
                        if hasattr(msg, 'content'):
                            text_parts.append(f"[{msg.role}]: {msg.content}")
                        elif hasattr(msg, 'blocks'):
                            for block in msg.blocks:
                                if hasattr(block, 'text'):
                                    text_parts.append(f"[{msg.role}]: {block.text}")
                    return "\n\n".join(text_parts) if text_parts else str(payload)
            
            # Fallback : convertir en string simple
            return str(payload)
        except Exception as e:
            logger.warning(f"Erreur extraction prompt: {e}")
            return str(payload)
    
    def _check_graph_exists(self) -> bool:
        """Vérifie si un graphe existe déjà (Neo4j ou SimpleGraphStore)."""
        if self.use_neo4j and NEO4J_AVAILABLE:
            try:
                # Vérifier si Neo4j a des nœuds
                uri = "bolt://localhost:7687"
                user = "neo4j"
                password = "password"
                
                driver = GraphDatabase.driver(uri, auth=(user, password))
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                    record = result.single()
                    count = record["count"] if record else 0
                driver.close()
                
                exists = count > 0
                print(f"   Neo4j: {'✅ Graphe trouvé' if exists else '❌ Graphe vide'}")
                return exists
            except Exception as e:
                logger.warning(f"Impossible de vérifier Neo4j: {e}")
                return False
        else:
            # Pour SimpleGraphStore, vérifier si le fichier de store existe
            graph_store_path = os.path.join(self.persist_dir, "graph_store.json")
            exists = os.path.exists(graph_store_path)
            print(f"   SimpleGraphStore: {'✅ Données trouvées' if exists else '❌ Pas de données'}")
            return exists
    
    def _load_existing_indexes(self) -> None:
        """Charge les index existants (vectoriel et graphe) depuis le stockage."""
        try:
            # Créer un contexte de stockage qui inclut notre graph_store
            if self.use_neo4j and NEO4J_AVAILABLE:
                # Recréer la connexion Neo4j
                print(f"🕸️ Reconnexion à Neo4j...")
                self.graph_store = Neo4jGraphStore(
                    username="neo4j",
                    password="password",
                    url="bolt://localhost:7687",
                    database="neo4j",
                )
                storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
            else:
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            
            print("   → Chargement de l'index VECTORIEL...")
            self.vector_index = load_index_from_storage(storage_context, index_id="vector_index")
            
            print("   → Chargement de l'index GRAPHE...")
            self.graph_index = load_index_from_storage(storage_context, index_id="graph_index")
            
            print("✅ Index chargés avec succès.")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des index: {e}")
            print(f"❌ Erreur: {e}")
            print("   → Reconstruction complète nécessaire...")
            self.force_rebuild = True
            self.load_and_index()

    def generate_graph_image(self, output_file="graph.png"):
        """Génère une image PNG du graphe (fonctionne avec Neo4j ET SimpleGraphStore)."""
        print(f"\n🎨 Génération de l'image du graphe vers {output_file}...")
        
        try:
            G = nx.DiGraph()
            
            if self.use_neo4j and NEO4J_AVAILABLE:
                # Récupérer depuis Neo4j
                uri = "bolt://localhost:7687"
                user = "neo4j"
                password = "password"
                
                if hasattr(self.graph_store, 'url'): uri = self.graph_store.url
                if hasattr(self.graph_store, 'username'): user = self.graph_store.username
                if hasattr(self.graph_store, 'password'): password = self.graph_store.password

                driver = GraphDatabase.driver(uri, auth=(user, password))
                
                with driver.session() as session:
                    result = session.run("MATCH (n)-[r]->(m) RETURN n.id as source, type(r) as relation, m.id as target LIMIT 100")
                    
                    count = 0
                    for record in result:
                        count += 1
                        source = record["source"]
                        target = record["target"]
                        relation = record["relation"]
                        
                        G.add_edge(source, target, label=relation)
                    
                    if count == 0:
                        print("⚠️ Aucune relation trouvée dans Neo4j pour le dessin.")
                        return
                
                driver.close()
            else:
                # Récupérer depuis SimpleGraphStore
                if not hasattr(self.graph_index, 'graph_store') or self.graph_index.graph_store is None:
                    print("⚠️ Aucun graph store disponible.")
                    return
                
                graph_store = self.graph_index.graph_store
                
                # Récupérer tous les triplets du graphe
                if hasattr(graph_store, 'data') and hasattr(graph_store.data, 'edges'):
                    # SimpleGraphStore structure
                    count = 0
                    for source_id, targets in graph_store.data.edges.items():
                        for target_id, relations in targets.items():
                            for relation in relations:
                                count += 1
                                G.add_edge(source_id, target_id, label=relation)
                                # Limiter pour éviter les graphes trop énormes
                                if count > 100:
                                    break
                            if count > 100:
                                break
                        if count > 100:
                            break
                    
                    if count == 0:
                        print("⚠️ Aucune relation trouvée dans SimpleGraphStore pour le dessin.")
                        return
                else:
                    print("⚠️ Structure de SimpleGraphStore non reconnue.")
                    return
            
            # Dessiner le graphe
            if G.number_of_nodes() > 0:
                plt.figure(figsize=(14, 10))
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
                nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                        node_size=2000, font_size=7, font_weight='bold', 
                        arrows=True, edge_color='gray', arrowsize=15)
                
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
                
                plt.title("Graphe de Connaissances")
                plt.tight_layout()
                
                # Sauvegarder avec chemin absolu pour être sûr
                abs_output = os.path.abspath(output_file)
                plt.savefig(abs_output, dpi=150)
                plt.close()
                print(f"✅ Image sauvegardée : {abs_output}")
                
                # Vérifier que le fichier existe vraiment
                if os.path.exists(abs_output):
                    print(f"   → Fichier confirmé : {os.path.getsize(abs_output)} octets")
                else:
                    print(f"   ⚠️ ATTENTION : Le fichier n'a pas été créé !")
            else:
                print("⚠️ Le graphe est vide, aucune image générée.")
                
        except Exception as e:
            print(f"❌ Erreur lors de la génération de l'image : {e}")

    def query(self, question: str):
        """Pose une question aux deux systèmes et compare."""
        print(f"\n{'='*60}")
        print(f"❓ QUESTION : {question}")
        print(f"{'='*60}")
        
        # Capture de l'état des logs avant la requête
        start_idx = len(self.llama_debug.get_llm_inputs_outputs())
        
        # 1. Recherche Vectorielle
        print("\n--- 🔍 Approche VECTORIELLE ---")
        vector_engine = self.vector_index.as_query_engine(similarity_top_k=self.top_k)
        response_vec = vector_engine.query(question)
        
        print(f"Sources utilisées ({len(response_vec.source_nodes)} chunks) :")
        for node in response_vec.source_nodes:
            print(f"  - [Score {node.score:.2f}] {node.node.get_content()[:100]}...")
            
        print(f"\n💬 RÉPONSE VECTORIELLE :\n{response_vec}")
        
        # Capture du prompt Vectoriel
        current_events = self.llama_debug.get_llm_inputs_outputs()
        vec_prompt = "Pas de prompt trouvé (peut-être pas d'appel LLM ?)"
        if len(current_events) > start_idx:
             # On prend le dernier événement LLM
             last_event_start = current_events[-1][0]
             last_payload = last_event_start.payload
             vec_prompt = self._extract_prompt_from_payload(last_payload)
        
        # Mise à jour de l'index pour la suite
        start_idx = len(current_events)

        # 2. Recherche Graphe avec Retriever Hybride
        print("\n--- 🕸️ Approche GRAPHE (Hybride: Triplets + Texte) ---")
        
        # Créer le retriever hybride qui combine graphe et vecteur
        hybrid_retriever = HybridGraphRetriever(
            graph_index=self.graph_index,
            vector_index=self.vector_index,
            similarity_top_k=self.top_k
        )
        
        # Créer un query engine avec le retriever hybride
        graph_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            response_mode="compact",
        )
        
        response_graph = graph_engine.query(question)
        
        print(f"Sources utilisées (Texte + Triplets du graphe) :")
        # LlamaIndex retourne des noeuds combinés
        for i, node in enumerate(response_graph.source_nodes):
             content = node.node.get_content()
             label = "TRIPLET" if ("->" in content or "(" in content) else "TEXTE"
             print(f"  [{label}] {content[:150]}...")

        print(f"\n💬 RÉPONSE GRAPHE :\n{response_graph}")
        
        # Capture du prompt Graphe
        current_events = self.llama_debug.get_llm_inputs_outputs()
        graph_prompts = []
        if len(current_events) > start_idx:
             # On récupère TOUS les événements depuis le début de la requête graphe
             for i in range(start_idx, len(current_events)):
                 event = current_events[i][0] # event_start
                 payload = event.payload
                 p = self._extract_prompt_from_payload(payload)
                 graph_prompts.append(f"--- Event {i-start_idx+1} ---\n{p}\n")
        
        graph_prompt = "\n".join(graph_prompts) if graph_prompts else "Pas de prompt trouvé"

        # Sauvegarde dans prompt.txt
        try:
            with open("prompt.txt", "w", encoding="utf-8") as f:
                f.write(f"QUESTION: {question}\n\n")
                f.write("="*40 + "\n")
                f.write("APPROCHE VECTORIELLE (RAG Classique)\n")
                f.write("="*40 + "\n")
                f.write(vec_prompt + "\n\n")
                f.write("="*40 + "\n")
                f.write("APPROCHE GRAPHE (Graph RAG)\n")
                f.write("="*40 + "\n")
                f.write(graph_prompt + "\n")
            print(f"\n📝 Prompts sauvegardés dans prompt.txt")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde prompt.txt: {e}")
        
        # Comparaison (visuelle pour l'utilisateur)
        print("\n" + "-"*60)

    def interactive_loop(self):
        print("\nMode interactif. Tapez 'q' pour quitter.")
        while True:
            q = input("\nVotre question : ")
            if q.lower() in ['q', 'quit', 'exit']:
                break
            self.query(q)

def main():
    parser = argparse.ArgumentParser(description="Demo RAG Graph vs Vector")
    parser.add_argument("--data", default="data", help="Dossier de données")
    parser.add_argument("--provider", help="Nom du provider LLM (ex: MISTRAL_NEMO)")
    parser.add_argument("--neo4j", action="store_true", help="Utiliser Neo4j")
    parser.add_argument("--top-k", type=int, default=3, help="Nombre de chunks à récupérer (défaut: 3)")
    parser.add_argument("--reload", action="store_true", help="Recharger les index existants si possible")
    parser.add_argument("--no-interactive", action="store_true", help="Passer le mode interactif")
    args = parser.parse_args()
    
    # Sélection automatique du provider si non spécifié
    provider = args.provider or os.getenv("LLM_PROVIDER")
    
    demo = GraphRAGDemo(
        data_dir=args.data,
        provider_name=provider,
        use_neo4j=args.neo4j,
        top_k=args.top_k,
        reload=args.reload
    )
    
    # Utiliser le flag reload pour forcer la reconstruction si demandé
    if args.reload:
        demo.force_rebuild = True
    
    demo.load_and_index()
    
    # Questions de démonstration
    print("\n" + "="*60)
    print("DÉMONSTRATION AUTOMATIQUE")
    print("="*60)
    
    demo.query("Qui sont les personnages principaux ?")
    
    # Mode interactif (sauf si --no-interactive)
    if not args.no_interactive:
        demo.interactive_loop()
    else:
        print("\n✅ Démonstration terminée (mode interactif désactivé).")

if __name__ == "__main__":
    main()
