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

# ============================================================================
# IMPORTS STANDARDS
# ============================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import logging
import argparse
from typing import Optional
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================================
# IMPORTS LLAMAINDEX
# ============================================================================
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.prompts import PromptTemplate
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

# ============================================================================
# IMPORTS PERSONNALISÉS
# ============================================================================
from entity_normalizer import EntityNormalizer
from triplet_extractor import TripletExtractor
from prompt_extractor import PromptExtractor
from neo4j_manager import Neo4jManager
import config

# ============================================================================
# CONFIGURATION
# ============================================================================
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Import du gestionnaire de providers existant
try:
    from llm_providers import get_provider, PROVIDERS
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from llm_providers import get_provider, PROVIDERS

# Tentative d'import de Neo4j (optionnel)
try:
    from llama_index.graph_stores.neo4j import Neo4jGraphStore
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class GraphRAGDemo:
    """
    Démonstrateur complet du RAG Graph avec LlamaIndex.
    
    Combine plusieurs techniques :
    - Index vectoriel classique pour la similarité de texte
    - Index graphe pour les relations entre entités
    - Fusion des deux approches (QueryFusionRetriever)
    - Reranking pour optimiser les résultats
    """

    def __init__(
        self,
        data_dir: str = config.DEFAULT_DATA_DIR,
        provider_name: Optional[str] = None,
        use_neo4j: bool = False,
        top_k: int = 7,
        reload: bool = False,
        persist_dir: str = config.DEFAULT_PERSIST_DIR,
    ):
        """
        Initialise le démonstrateur.
        
        Args:
            data_dir: Dossier contenant les documents à indexer
            provider_name: Nom du provider LLM (ex: MISTRAL_NEMO)
            use_neo4j: Utiliser Neo4j au lieu de SimpleGraphStore
            top_k: Nombre de chunks à récupérer (défaut: 7)
            reload: Forcer la reconstruction des index
            persist_dir: Dossier de stockage des index
        """
        self.data_dir = data_dir
        self.top_k = top_k
        self.persist_dir = persist_dir
        self.force_rebuild = reload

        # 1. Configuration du LLM via le provider existant
        provider_cfg = get_provider(name=provider_name)
        logger.info(f"🔌 Provider LLM : {provider_cfg.name} ({provider_cfg.model})")

        api_base = provider_cfg.url.replace("/chat/completions", "")
        api_key = provider_cfg.api_key() or "fake-key"

        self.llm = OpenAILike(
            model=provider_cfg.model,
            api_base=api_base,
            api_key=api_key,
            is_chat_model=True,
            context_window=config.LLM_CONTEXT_WINDOW,
            max_tokens=config.LLM_MAX_TOKENS,
            timeout=config.LLM_TIMEOUT,
        )

        # 2. Configuration de l'Embedding
        logger.info(f"🧠 Chargement du modèle d'embedding...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL
        )

        # Appliquer la configuration globale
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = config.CHUNK_SIZE

        # Configuration du Callback Manager
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=False)
        callback_manager = CallbackManager([self.llama_debug])
        Settings.callback_manager = callback_manager
        
        # Initialiser les modules utilitaires
        self.entity_normalizer = EntityNormalizer()
        self.triplet_extractor = TripletExtractor(self.llm)
        self.prompt_extractor = PromptExtractor()
        self.neo4j_manager = Neo4jManager() if use_neo4j else None

        # 3. Configuration du Graph Store
        self.use_neo4j = use_neo4j
        if use_neo4j:
            if not NEO4J_AVAILABLE:
                logger.warning("Neo4j demandé mais librairie manquante. Fallback sur SimpleGraphStore.")
                self.graph_store = SimpleGraphStore()
            else:
                # Tentative de connexion dans tous les cas
                if self.neo4j_manager.connect():
                    logger.info("✅ Connexion réussie à Neo4j.")
                    # Si on ne recharge pas (donc on suppose un usage existant) et qu'on veut repartir de zéro,
                    # le nettoyage se fera plus tard dans load_and_index si force_rebuild est True.
                    # Ici on initialise juste le store.
                    self.graph_store = Neo4jGraphStore(
                        username=config.NEO4J_USER,
                        password=config.NEO4J_PASSWORD,
                        url=config.NEO4J_URL,
                        database=config.NEO4J_DATABASE,
                    )
                else:
                    logger.warning("Impossible de se connecter à Neo4j. Fallback sur SimpleGraphStore.")
                    self.graph_store = SimpleGraphStore()
        else:
            logger.info("💾 Utilisation du stockage de graphe en mémoire (SimpleGraphStore).")
            self.graph_store = SimpleGraphStore()

        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store)

        self.vector_index = None
        self.graph_index = None

    def load_and_index(self):
        """
        Charge les documents et construit les index.
        
        Stratégie :
        - Vérifie si les index existent et les recharge si possible
        - Sinon, construit les index depuis zéro
        - Supporte la reconstruction forcée via self.force_rebuild
        """

        if not self.force_rebuild:
            if self._check_graph_exists() and os.path.exists(self.persist_dir):
                logger.info("✅ Index existants détectés. Chargement depuis le stockage...")
                self._load_existing_indexes()
                return
        else:
            logger.info("🔄 Reconstruction forcée demandée.")

        logger.info(f"📂 Chargement des documents depuis {self.data_dir}...")
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        logger.info(f"   → {len(documents)} documents chargés.")

        # Nettoyage Neo4j avant reconstruction
        if self.use_neo4j and self.neo4j_manager and self.neo4j_manager.is_connected():
            self.neo4j_manager.clear_database()

        # Construction de l'index vectoriel
        logger.info("🚀 Construction de l'index VECTORIEL...")
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=True
        )
        self.vector_index.set_index_id("vector_index")

        # Construction de l'index graphe
        logger.info("🕸️ Construction de l'index GRAPHE (Extraction & Normalisation)...")
        logger.info("   (Ceci peut prendre du temps car le LLM doit extraire les entités)")

        def custom_extract_and_normalize_triplets(text):
            """Fonction customisée pour extraire et normaliser les triplets."""
            # 1. Extraction via LLM
            raw_triplets = self.triplet_extractor.extract_raw_triplets(text)

            # 2. Normalisation des entités
            normalized_triplets = []
            logger.debug("🔍 Traitement d'un chunk :")
            for subj, pred, obj in raw_triplets:
                norm_subj = self.entity_normalizer.normalize(subj)
                norm_obj = self.entity_normalizer.normalize(obj)
                
                if TripletExtractor.validate_triplet(norm_subj, pred, norm_obj):
                    normalized_triplets.append((norm_subj, pred, norm_obj))
                    logger.debug(f"   • {norm_subj} -> {pred} -> {norm_obj}")
            
            return normalized_triplets

        self.graph_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=config.MAX_TRIPLETS_PER_CHUNK,
            include_embeddings=True,
            show_progress=True,
            kg_triplet_extract_fn=custom_extract_and_normalize_triplets,
        )
        self.graph_index.set_index_id("graph_index")

        logger.info("✅ Indexation terminée.")

        # Afficher les statistiques de normalisation
        stats = self.entity_normalizer.get_statistics()
        logger.info(f"📊 Normalisation : {stats['canonical_entities']} entités canoniques, "
                   f"{stats['total_mentions']} mentions totales")

        # Sauvegarde
        logger.info(f"💾 Sauvegarde des index dans {self.persist_dir}...")
        self.storage_context.persist(persist_dir=self.persist_dir)
        logger.info("   → Sauvegarde terminée.")

        self.generate_graph_image()

    def _check_graph_exists(self) -> bool:
        """Vérifie si un graphe existe déjà."""
        if self.use_neo4j and self.neo4j_manager and self.neo4j_manager.is_connected():
            return self.neo4j_manager.graph_exists()
        else:
            graph_store_path = os.path.join(self.persist_dir, "graph_store.json")
            return os.path.exists(graph_store_path)

    def _load_existing_indexes(self) -> None:
        """
        Charge les index existants (vectoriel et graphe) depuis le stockage.
        
        Gère les deux backends : Neo4j et SimpleGraphStore.
        """
        try:
            if self.use_neo4j and self.neo4j_manager:
                if not self.neo4j_manager.is_connected():
                    self.neo4j_manager.connect()
                
                if self.neo4j_manager.is_connected():
                    logger.info("🕸️ Reconnexion à Neo4j...")
                    self.graph_store = Neo4jGraphStore(
                        username=config.NEO4J_USER,
                        password=config.NEO4J_PASSWORD,
                        url=config.NEO4J_URL,
                        database=config.NEO4J_DATABASE,
                    )
                    storage_context = StorageContext.from_defaults(
                        graph_store=self.graph_store,
                        persist_dir=self.persist_dir)
                else:
                    logger.warning("Impossible de se connecter à Neo4j, utilisation du stockage local")
                    storage_context = StorageContext.from_defaults(
                        persist_dir=self.persist_dir)
            else:
                storage_context = StorageContext.from_defaults(
                    persist_dir=self.persist_dir)

            logger.info("   → Chargement de l'index VECTORIEL...")
            self.vector_index = load_index_from_storage(
                storage_context, index_id="vector_index")

            logger.info("   → Chargement de l'index GRAPHE...")
            self.graph_index = load_index_from_storage(
                storage_context, index_id="graph_index")

            logger.info("✅ Index chargés avec succès.")

        except Exception as e:
            logger.error(f"Erreur lors du chargement des index: {e}")
            logger.info("   → Reconstruction complète nécessaire...")
            self.force_rebuild = True
            self.load_and_index()

    def generate_graph_image(self, output_file: str = "graph.png"):
        """
        Génère une image PNG du graphe de connaissances.
        
        Visualise les nœuds et relations du graphe de manière optimisée.
        
        Args:
            output_file: Chemin de sortie du fichier PNG
        """
        logger.info(f"🎨 Génération de l'image du graphe vers {output_file}...")
        try:
            G = nx.DiGraph()
            
            # Extraction des relations selon le backend
            if self.use_neo4j and self.neo4j_manager and self.neo4j_manager.is_connected():
                self._load_graph_from_neo4j(G)
            else:
                self._load_graph_from_simple_store(G)
            
            # Rendu du graphe
            if G.number_of_nodes() > 0:
                self._render_graph(G, output_file)
            else:
                logger.warning("⚠️ Le graphe est vide.")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de l'image : {e}")

    def _load_graph_from_neo4j(self, G: nx.DiGraph):
        """Charge les données du graphe depuis Neo4j."""
        try:
            if self.neo4j_manager.driver:
                with self.neo4j_manager.driver.session(database=config.NEO4J_DATABASE) as session:
                    result = session.run(
                        "MATCH (n)-[r]->(m) RETURN n.id as source, type(r) as relation, m.id as target LIMIT 100")
                    for record in result:
                        G.add_edge(record["source"], record["target"], label=record["relation"])
        except Exception as e:
            logger.warning(f"Erreur chargement Neo4j: {e}")

    def _load_graph_from_simple_store(self, G: nx.DiGraph):
        """Charge les données du graphe depuis SimpleGraphStore."""
        try:
            if hasattr(self.graph_index, 'graph_store') and self.graph_index.graph_store:
                graph_store = self.graph_index.graph_store
                if hasattr(graph_store, '_data') and hasattr(graph_store._data, 'graph_dict'):
                    count = 0
                    for source_id, relations_list in graph_store._data.graph_dict.items():
                        for item in relations_list:
                            if len(item) >= 2:
                                relation = item[0]
                                target_id = item[1]
                                G.add_edge(source_id, target_id, label=relation)
                                count += 1
                                if count > 100:
                                    break
                        if count > 100:
                            break
        except Exception as e:
            logger.warning(f"Erreur chargement SimpleGraphStore: {e}")

    def _render_graph(self, G: nx.DiGraph, output_file: str):
        """Rend et sauvegarde le graphe en image."""
        # Configuration de la figure
        plt.figure(figsize=(24, 18))
        pos = nx.spring_layout(G, k=2.5, iterations=200, seed=42, scale=2.0)

        # Éléments visuels
        nx.draw_networkx_nodes(G, pos, node_color='#A0CBE2', node_size=3000, alpha=0.9,
                             edgecolors='white', linewidths=2)
        
        nx.draw_networkx_edges(G, pos, edge_color='#888888', arrows=True, arrowsize=25, 
                             width=1.5, connectionstyle='arc3,rad=0.15', 
                             min_source_margin=15, min_target_margin=15)
        
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='black',
                              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=3.0))

        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5,
            bbox=dict(facecolor='#EEEEEE', edgecolor='none', alpha=0.8, pad=1.0)
        )
        
        plt.title("Graphe de Connaissances (Vue Optimisée)", fontsize=20)
        plt.axis('off')
        plt.tight_layout()
        
        abs_output = os.path.abspath(output_file)
        plt.savefig(abs_output, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Image sauvegardée : {abs_output}")

    def query(self, question: str):
        """
        Pose une question aux deux systèmes et compare les résultats.
        
        Approches :
        1. Recherche vectorielle classique
        2. Approche graphe optimisée (fusion + reranking)
        
        Args:
            question: La question à poser
        """
        logger.info("="*60)
        logger.info(f"❓ QUESTION : {question}")
        logger.info("="*60)

        start_idx = len(self.llama_debug.get_llm_inputs_outputs())

        # 1. Recherche Vectorielle
        logger.info("--- 🔍 Approche VECTORIELLE ---")
        vector_engine = self.vector_index.as_query_engine(
            similarity_top_k=self.top_k)
        response_vec = vector_engine.query(question)

        logger.info(f"Sources utilisées ({len(response_vec.source_nodes)} chunks) :")
        for node in response_vec.source_nodes:
            content_preview = node.node.get_content()[:100].replace('\n', ' ')
            logger.info(f"  - [Score {node.score:.2f}] {content_preview}...")
        logger.info(f"💬 RÉPONSE VECTORIELLE :\n{response_vec}")

        # Capture prompt vectoriel
        current_events = self.llama_debug.get_llm_inputs_outputs()
        logger.info(f"DEBUG: Events count after Vector: {len(current_events)} (Start: {start_idx})")
        vec_prompt = "Pas de prompt trouvé"
        
        if len(current_events) > start_idx:
            # Fix: Search backwards for the first valid event with payload
            # Sometimes the very last event might be a generic event without payload
            for i in range(len(current_events) - 1, start_idx - 1, -1):
                event = current_events[i]
                # Fix: Handle tuple/list (legacy/container) and object (new) formats
                if isinstance(event, (tuple, list)) and len(event) > 0:
                    event_obj = event[0]
                else:
                    event_obj = event
                
                if hasattr(event_obj, 'payload'):
                    extracted = self.prompt_extractor.extract_from_payload(event_obj.payload)
                    if extracted and "Pas de prompt trouvé" not in extracted:
                         vec_prompt = extracted
                         logger.info(f"DEBUG: Found Vector prompt in event index {i} (type: {type(event_obj)})")
                         break
                else:
                     logger.debug(f"DEBUG: Event at index {i} skipped (no payload)")


        start_idx = len(current_events)

        # 2. Recherche Graphe Optimisée
        logger.info("--- 🕸️ Approche GRAPHE Optimisée (Fusion + Reranking) ---")

        # Retrievers
        graph_retriever = KnowledgeGraphRAGRetriever(
            storage_context=self.storage_context,
            include_text=True,
            verbose=True,
            graph_traversal_depth=config.GRAPH_TRAVERSAL_DEPTH,
            max_knowledge_sequence=config.MAX_KNOWLEDGE_SEQUENCE,
        )

        vector_retriever = self.vector_index.as_retriever(
            similarity_top_k=self.top_k
        )

        # Fusion
        logger.info("   → Fusion des résultats Graph + Vector...")
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, graph_retriever],
            similarity_top_k=self.top_k * 2,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            verbose=True,
        )

        # Reranking
        logger.info("   → Reranking des résultats...")
        reranker = SentenceTransformerRerank(
            model=config.RERANKER_MODEL,
            top_n=self.top_k,
        )

        # Query Engine avec prompt spécialisé
        graph_engine = RetrieverQueryEngine.from_args(
            retriever=fusion_retriever,
            node_postprocessors=[reranker],
            text_qa_template=PromptTemplate(config.GRAPH_QA_PROMPT),
            response_mode="tree_summarize",
            verbose=True
        )

        response_graph = graph_engine.query(question)

        # Analyse des sources
        logger.info("Sources finales utilisées (après fusion et reranking) :")
        for node in response_graph.source_nodes:
            content = node.node.get_content()[:100].replace('\n', ' ')
            source_type = "TEXTE"
            meta = node.node.metadata or {}
            if "relationship" in str(meta) or "triplet" in str(meta) or "->" in content:
                source_type = "GRAPHE"
            logger.info(f"  [{source_type}] [Score {node.score:.2f}] {content}...")

        if not response_graph.response or not str(response_graph).strip():
            logger.warning(f"⚠️ RÉPONSE VIDE DÉTECTÉE. Contenu brut : {repr(response_graph)}")
        
        logger.info(f"💬 RÉPONSE GRAPHE :\n{response_graph.response}")

        # Capture prompts Graphe
        current_events = self.llama_debug.get_llm_inputs_outputs()
        logger.info(f"DEBUG: Events count after Graph: {len(current_events)} (Start: {start_idx})")
        graph_prompts = []
        if len(current_events) > start_idx:
            for i in range(start_idx, len(current_events)):
                event = current_events[i]
                if isinstance(event, (tuple, list)) and len(event) > 0:
                    event_obj = event[0]
                else:
                    event_obj = event
                
                # Extraire le payload
                payload = getattr(event_obj, 'payload', None)
                
                if payload:
                    p = self.prompt_extractor.extract_from_payload(payload)
                    graph_prompts.append(f"--- Event {i-start_idx+1} ---\n{p}\n")
        
        graph_prompt = "\n".join(graph_prompts) if graph_prompts else "Pas de prompt trouvé"

        # Sauvegarde des prompts
        self._save_prompts(question, vec_prompt, graph_prompt)

        logger.info("-"*60)

    def _save_prompts(self, question: str, vec_prompt: str, graph_prompt: str):
        """Sauvegarde les prompts dans un fichier pour analyse."""
        try:
            with open("prompt.txt", "w", encoding="utf-8") as f:
                f.write(f"QUESTION: {question}\n\n")
                f.write("="*40 + "\n")
                f.write("APPROCHE VECTORIELLE\n")
                f.write("="*40 + "\n")
                f.write(vec_prompt + "\n\n")
                f.write("="*40 + "\n")
                f.write("APPROCHE GRAPHE OPTIMISEE\n")
                f.write("="*40 + "\n")
                f.write(graph_prompt + "\n")
            logger.info("📝 Prompts sauvegardés dans prompt.txt")
        except Exception as e:
            logger.warning(f"⚠️ Erreur sauvegarde prompt.txt: {e}")

    def interactive_loop(self):
        """
        Lance une boucle interactive pour poser des questions.
        
        Permet à l'utilisateur de poser plusieurs questions et d'explorer
        les résultats du RAG Graph.
        """
        logger.info("Mode interactif. Tapez 'q' pour quitter.")
        while True:
            q = input("\nVotre question : ").strip()
            if q.lower() in ['q', 'quit', 'exit']:
                break
            if q:
                self.query(q)


    def close(self):
        """Ferme les connexions."""
        # 1. Fermer notre manager personnalisé
        if self.use_neo4j and self.neo4j_manager:
            self.neo4j_manager.close()
            
        # 2. Fermer le graph store de LlamaIndex
        if self.use_neo4j and self.graph_index:
            try:
                # Accès direct au store souvent nécessaire car LlamaIndex n'expose pas toujours close()
                store = self.graph_index.graph_store
                if hasattr(store, 'close'):
                    store.close()
                elif hasattr(store, '_driver'):
                    store._driver.close()
                    logger.debug("Driver Neo4jGraphStore fermé manuellement.")
            except Exception as e:
                logger.warning(f"Erreur fermeture Neo4jGraphStore: {e}")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Entrée principale du programme."""
    parser = argparse.ArgumentParser(
        description="Demo RAG Graph vs Vector - Compare les deux approches"
    )
    parser.add_argument(
        "--data",
        default=config.DEFAULT_DATA_DIR,
        help="Dossier de données"
    )
    parser.add_argument(
        "--provider",
        help="Nom du provider LLM (ex: MISTRAL_NEMO)"
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Utiliser Neo4j pour le stockage de graphe"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=7,
        help="Nombre de chunks à récupérer (défaut: 7)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Recharger les index existants si possible"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Passer le mode interactif"
    )
    
    args = parser.parse_args()

    provider = args.provider or os.getenv("LLM_PROVIDER")

    demo = GraphRAGDemo(
        data_dir=args.data,
        provider_name=provider,
        use_neo4j=args.neo4j,
        top_k=args.top_k,
        reload=args.reload
    )

    try:
        if args.reload:
            demo.force_rebuild = True

        demo.load_and_index()

        logger.info("="*60)
        logger.info("DÉMONSTRATION AUTOMATIQUE")
        logger.info("="*60)

        demo.query("Pourquoi se méfier de Jules ?")

        if not args.no_interactive:
            demo.interactive_loop()
        else:
            logger.info("✅ Démonstration terminée (mode interactif désactivé).")
            
    finally:
        if demo:
            demo.close()


if __name__ == "__main__":
    main()

