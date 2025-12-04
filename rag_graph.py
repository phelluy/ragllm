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
import matplotlib.pyplot as plt
from neo4j import GraphDatabase

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
    """Retriever hybride qui utilise le graphe pour découvrir des chunks pertinents.
    
    Stratégie (Option 2 - Graphe autonome avec expansion):
    1. Le graphe identifie les triplets pertinents à la requête
    2. On récupère les IDs des chunks sources de ces triplets
    3. On charge le texte complet de ces chunks depuis l'index vectoriel
    4. On retourne les chunks complets avec leurs triplets associés
    
    Cela garantit que chaque triplet est accompagné de son contexte textuel d'origine.
    """
    
    def __init__(self, graph_index, vector_index, similarity_top_k: int = 3):
        self.graph_index = graph_index
        self.vector_index = vector_index
        self.similarity_top_k = similarity_top_k
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Récupère les chunks via le graphe, puis charge leur texte source complet."""
        
        # 1. Récupérer les triplets pertinents du graphe
        graph_retriever = self.graph_index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            include_text=True,  # Inclure le texte source si disponible
        )
        graph_nodes = graph_retriever.retrieve(query_bundle)
        
        # 2. Extraire les IDs des chunks sources des triplets
        source_node_ids = set()
        triplet_nodes = []
        
        for node in graph_nodes:
            # Les triplets ont souvent un ref_doc_id qui pointe vers le chunk source
            if hasattr(node.node, 'ref_doc_id') and node.node.ref_doc_id:
                source_node_ids.add(node.node.ref_doc_id)
            
            # Garder aussi les triplets pour information structurée
            content = node.node.get_content()
            if "->" in content or "(" in content:
                triplet_nodes.append(node)
        
        # 3. Récupérer les chunks sources complets depuis le docstore
        all_nodes = []
        docstore = self.vector_index.docstore
        
        for node_id in source_node_ids:
            try:
                # Récupérer le noeud complet avec son texte
                source_node = docstore.get_node(node_id)
                if source_node:
                    # Créer un NodeWithScore pour maintenir la cohérence
                    # Score = moyenne des scores des triplets issus de ce chunk
                    related_triplet_scores = [
                        n.score for n in graph_nodes 
                        if hasattr(n.node, 'ref_doc_id') and n.node.ref_doc_id == node_id
                    ]
                    avg_score = sum(related_triplet_scores) / len(related_triplet_scores) if related_triplet_scores else 0.5
                    
                    all_nodes.append(NodeWithScore(node=source_node, score=avg_score))
            except Exception as e:
                logger.warning(f"Impossible de récupérer le chunk {node_id}: {e}")
        
        # 4. Ajouter les triplets comme contexte structuré supplémentaire
        # (optionnel, pour que le LLM voie aussi les relations extraites)
        all_nodes.extend(triplet_nodes)
        
        # 5. Si aucun chunk source trouvé via le graphe, fallback sur recherche vectorielle
        if not all_nodes:
            logger.info("Aucun chunk trouvé via le graphe, fallback sur recherche vectorielle")
            vector_retriever = self.vector_index.as_retriever(
                similarity_top_k=self.similarity_top_k
            )
            all_nodes = vector_retriever.retrieve(query_bundle)
        
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
        self.reload = reload
        self.persist_dir = persist_dir
        
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
        """Charge les documents et construit les index (ou les recharge)."""
        
        # Si reload demandé et que le dossier existe
        if self.reload and os.path.exists(self.persist_dir):
            print(f"\n♻️  Rechargement des index depuis {self.persist_dir}...")
            try:
                # Recharger le contexte de stockage
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                
                # Recharger les index
                # Note: load_index_from_storage charge un index. Si on en a plusieurs, il faut les distinguer.
                # Ici on va tricher un peu en essayant de charger par ID si possible, ou juste charger ce qu'on trouve.
                # Pour simplifier dans cette démo : on suppose que load_index_from_storage va nous rendre l'index vectoriel
                # si c'est le seul persisté ou le principal.
                # Mais KnowledgeGraphIndex et VectorStoreIndex sont stockés ensemble.
                
                print("   → Chargement de l'index VECTORIEL...")
                self.vector_index = load_index_from_storage(storage_context, index_id="vector_index")
                
                print("   → Chargement de l'index GRAPHE...")
                self.graph_index = load_index_from_storage(storage_context, index_id="graph_index")
                
                # Si on utilise Neo4j, le graph_store doit être correctement reconnecté
                # Mais StorageContext.from_defaults(persist_dir) va charger un SimpleGraphStore depuis le disque
                # si on ne lui dit pas le contraire.
                # Si on veut Neo4j, il faut que le storage_context utilise NOTRE graph_store (Neo4j)
                # et qu'on ne charge que les docstores/indexstores depuis le disque.
                
                # Correction pour Neo4j :
                if self.use_neo4j and NEO4J_AVAILABLE:
                     # On garde notre graph_store Neo4j connecté
                     # On charge juste les autres stores (doc, index, vector)
                     # C'est un peu tricky avec LlamaIndex high-level.
                     # Le plus simple : re-créer le StorageContext avec notre graph_store,
                     # et charger les données persistées dedans.
                     pass 
                
                print("✅ Index rechargés avec succès.")
                return

            except Exception as e:
                print(f"⚠️ Erreur lors du rechargement : {e}")
                print("   → Bascule sur la régénération complète.")

        # Sinon (ou si échec), on génère tout
        print(f"\n📂 Chargement des documents depuis {self.data_dir}...")
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        print(f"   → {len(documents)} documents chargés.")
        
        # Construction de l'index Vectoriel
        print("\n🚀 Construction de l'index VECTORIEL...")
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context, # Important pour partager le même storage
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
        
        if self.use_neo4j and NEO4J_AVAILABLE:
             self.perform_entity_resolution()
             self.generate_graph_image()

    def perform_entity_resolution(self):
        """Fusionne les noeuds similaires (Entity Resolution basique)."""
        print("\n🔗 Exécution de la résolution d'entités (Fusion des noeuds 'Chevalière')...")
        # Récupération des paramètres (copié de generate_graph_image, à refactoriser idéalement)
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        if hasattr(self.graph_store, 'url'): uri = self.graph_store.url
        if hasattr(self.graph_store, 'username'): user = self.graph_store.username
        if hasattr(self.graph_store, 'password'): password = self.graph_store.password

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # Fusionner tous les noeuds contenant "Chevalière", "Bague" ou "Ring" (insensible à la casse)
                # On trie par longueur décroissante pour garder le nom le plus précis comme ID principal
                query = """
                MATCH (n:Entity)
                WHERE toLower(n.id) CONTAINS 'chevalière' OR toLower(n.id) CONTAINS 'bague' OR toLower(n.id) CONTAINS 'ring'
                WITH n ORDER BY size(n.id) DESC
                WITH collect(n) as nodes
                WHERE size(nodes) > 1
                CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true}) YIELD node
                RETURN count(node) as merged_count
                """
                result = session.run(query)
                record = result.single()
                if record:
                    print(f"   → {record['merged_count']} groupe(s) de noeuds fusionnés.")
                else:
                    print("   → Aucun noeud à fusionner.")
            driver.close()
        except Exception as e:
            print(f"⚠️ Erreur lors de la résolution d'entités : {e}")

    def generate_graph_image(self, output_file="graphneo4j.png"):
        """Génère une image PNG du graphe stocké dans Neo4j."""
        print(f"\n🎨 Génération de l'image du graphe vers {output_file}...")
        
        # Récupération des paramètres de connexion depuis le graph_store ou les attributs
        # Ici on réutilise les valeurs par défaut ou passées au constructeur
        # Note: Dans une implémentation plus propre, on stockerait ces infos
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "password"
        
        # On essaie de récupérer les infos du graph_store si c'est un Neo4jGraphStore
        if hasattr(self.graph_store, 'url'):
             uri = self.graph_store.url
        if hasattr(self.graph_store, 'username'):
             user = self.graph_store.username
        if hasattr(self.graph_store, 'password'):
             password = self.graph_store.password

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Récupérer tous les noeuds et relations
                # Attention: sur un gros graphe, limiter le nombre de résultats !
                result = session.run("MATCH (n)-[r]->(m) RETURN n.id as source, type(r) as relation, m.id as target LIMIT 100")
                
                G = nx.DiGraph()
                
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

                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(G, k=0.5)
                
                nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                        node_size=2000, font_size=8, font_weight='bold', 
                        arrows=True, edge_color='gray')
                
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
                
                plt.title("Graphe de Connaissances (Extrait)")
                plt.savefig(output_file)
                plt.close()
                print(f"✅ Image sauvegardée : {output_file}")
                
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
             vec_prompt = last_payload.get("formatted_prompt", str(last_payload))
        
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
                 p = payload.get("formatted_prompt", str(payload))
                 graph_prompts.append(f"--- Event {i} ---\n{p}\n")
        
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
    
    demo.load_and_index()
    
    # Questions de démonstration
    print("\n" + "="*60)
    print("DÉMONSTRATION AUTOMATIQUE")
    print("="*60)
    
    demo.query("Qui sont les personnages principaux ?")
    
    # Mode interactif
    demo.interactive_loop()

if __name__ == "__main__":
    main()
