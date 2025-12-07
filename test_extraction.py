
import sys
import logging
from llama_index.llms.openai_like import OpenAILike
from triplet_extractor import TripletExtractor
import config

# Configuration minimale
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Texte contenant l'indice implicite
TEST_TEXT = """
Sophie retrouve son amoureux après le bal. Elle lui raconte que sa mère a été fortement affectée par le vol.
Alors qu'il caresse sa main pour la consoler, Sophie remarque à son doigt une bague qu'elle ne lui avait jamais vue :
un anneau d'argent avec une tête de serpent.
Sophie se fige. Elle a lu dans les journaux que ce symbole est la marque de reconnaissance des membres de La Griffe.
"""

def main():
    print("--- Test d'Extraction Avancée ---")
    
    # 1. Setup LLM (copié de rag_graph.py)
    # Assumons que les variables d'env ou config par défaut marchent
    from llm_providers import get_provider
    provider_cfg = get_provider() # Défaut
    
    llm = OpenAILike(
        model=provider_cfg.model,
        api_base=provider_cfg.url.replace("/chat/completions", ""),
        api_key=provider_cfg.api_key() or "fake-key",
        is_chat_model=True
    )
    
    extractor = TripletExtractor(llm)
    
    # 2. Extraction
    print(f"Texte :\n{TEST_TEXT}\n")
    print("Extraction en cours...")
    triplets = extractor.extract_raw_triplets(TEST_TEXT)
    
    print("\n--- Résultats ---")
    found_link = False
    for s, p, o in triplets:
        print(f"  • {s} -> {p} -> {o}")
        # Le texte utilise "amoureux" ou "il", pas "Jules". On accepte ces termes.
        s_lower = s.lower()
        if "jules" in s_lower or "amoureux" in s_lower or "il" == s_lower:
            if "griffe" in o.lower():
                found_link = True
    
    if found_link:
        print("\n✅ SUCCÈS : Lien implicite détecté !")
    else:
        print("\n❌ ÉCHEC : Pas de lien direct trouvé entre l'amoureux et La Griffe.")

if __name__ == "__main__":
    main()
