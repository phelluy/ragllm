"""
Configuration centralisée pour le projet RAG Graph.
"""

# ============================================================================
# CONFIGURATION DES MODÈLES
# ============================================================================

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENTITY_NORMALIZER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================================
# CONFIGURATION DES PARAMÈTRES DE RECHERCHE
# ============================================================================

# Seuil de similarité pour la normalisation d'entités (0.0 à 1.0)
ENTITY_SIMILARITY_THRESHOLD = 0.85

# Nombre de triplets à extraire par chunk
MAX_TRIPLETS_PER_CHUNK = 15

# Nombre de niveaux de profondeur à explorer dans le graphe
GRAPH_TRAVERSAL_DEPTH = 2

# Nombre maximum de relations à inclure dans une séquence
MAX_KNOWLEDGE_SEQUENCE = 30

# ============================================================================
# CONFIGURATION NEO4J
# ============================================================================

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

# ============================================================================
# CONFIGURATION LLAMAINDEX
# ============================================================================

CHUNK_SIZE = 512
LLM_CONTEXT_WINDOW = 4096
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT = 60.0  # Suffisant pour la plupart des modèles

# ============================================================================
# CONFIGURATION STOCKAGE
# ============================================================================

DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_DATA_DIR = "data"

# ============================================================================
# CONFIGURATION DE LOGGING
# ============================================================================

LOG_LEVEL = "INFO"

# ============================================================================
# PROMPTS
# ============================================================================

TRIPLET_EXTRACT_PROMPT = (
    "You are an expert Knowledge Graph extractor. Given the text below, extract up to 15 knowledge triplets "
    "in the form of (subject, predicate, object). follow these rules strictly:\n"
    "1. Avoid stopwords.\n"
    "2. Output MUST be in the format: (Subject, Predicate, Object)\n"
    "3. One triplet per line.\n"
    "\n"
    "CRITICAL INSTRUCTION:\n"
    "- Extract explicit relationships.\n"
    "- ALSO infer hidden/implicit relationships if context strongly suggests them "
    "(e.g. if someone has a specific symbol/ring of a group, they are likely 'member of' that group).\n"
    "\n"
    "EXAMPLES:\n"
    "Text: 'Alice works at Google and is Bob's sister.'\n"
    "Triplets:\n"
    "(Alice, works_at, Google)\n"
    "(Alice, is_sister_of, Bob)\n"
    "\n"
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Triplets:"
)

GRAPH_QA_PROMPT = (
    "Tu es un assistant expert. Réponds à la question en utilisant le contexte fourni.\n"
    "Le contexte contient des extraits de textes ET des relations issues d'un graphe de connaissances.\n"
    "- Les relations du graphe aident à comprendre les liens entre entités.\n"
    "- Combine les deux sources pour une réponse complète.\n"
    "- Si la réponse n'est pas explicitement écrite, déduis-la à partir des indices (comportements suspects, liens cachés, etc.).\n\n"
    "CONTEXTE :\n{context_str}\n\n"
    "QUESTION : {query_str}\n\n"
    "RÉPONSE :"
)
