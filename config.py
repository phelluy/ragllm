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
GRAPH_TRAVERSAL_DEPTH = 3

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
    "Tu es un expert en extraction de Graphes de Connaissances. À partir du texte ci-dessous, extrais jusqu'à 15 triplets "
    "sous la forme (sujet, prédicat, objet). Respecte strictement ces règles :\n"
    "1. Évite les mots d'arrêt (stopwords).\n"
    "2. La sortie DOIT être au format : (Sujet, Prédicat, Objet)\n"
    "3. Un triplet par ligne.\n"
    "4. Conserve la langue originale du texte pour les sujets, prédicats et objets.\n"
    "\n"
    "INSTRUCTION CRITIQUE :\n"
    "- Extrais les relations explicites.\n"
    "- Déduis AUSSI les relations cachées/implicites si le contexte le suggère fortement \n"
    "EXEMPLES :\n"
    "Texte : 'Alice travaille chez Google et est la sœur de Bob.'\n"
    "Triplets :\n"
    "(Alice, travaille_chez, Google)\n"
    "(Alice, est_soeur_de, Bob)\n"
    "\n"
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Triplets :"
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
