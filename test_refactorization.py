"""
Tests simples pour valider la refactorisation.
"""

import sys
import os

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Teste que tous les modules s'importent correctement."""
    print("🧪 Test 1: Imports...")
    try:
        import config
        from entity_normalizer import EntityNormalizer
        from triplet_extractor import TripletExtractor
        from prompt_extractor import PromptExtractor
        from neo4j_manager import Neo4jManager
        from rag_graph import GraphRAGDemo
        print("   ✅ Tous les imports réussis")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_config():
    """Teste que la configuration se charge."""
    print("🧪 Test 2: Configuration...")
    try:
        import config
        assert hasattr(config, 'EMBEDDING_MODEL')
        assert hasattr(config, 'ENTITY_SIMILARITY_THRESHOLD')
        assert hasattr(config, 'NEO4J_URL')
        assert hasattr(config, 'TRIPLET_EXTRACT_PROMPT')
        assert hasattr(config, 'GRAPH_QA_PROMPT')
        print(f"   ✅ Configuration valide")
        print(f"      - Modèle: {config.EMBEDDING_MODEL}")
        print(f"      - Seuil: {config.ENTITY_SIMILARITY_THRESHOLD}")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_triplet_parser():
    """Teste le parsing des triplets."""
    print("🧪 Test 3: Parsing des triplets...")
    try:
        from triplet_extractor import TripletExtractor
        
        # Créer une instance factice (pas besoin de LLM pour parser)
        class FakeLLM:
            pass
        
        extractor = TripletExtractor(FakeLLM())
        
        # Test avec format parenthèses
        text1 = "(Alice, knows, Bob)\n(Bob, works_at, Google)"
        triplets1 = extractor._parse_triplets(text1)
        assert len(triplets1) == 2, f"Attendu 2 triplets, obtenu {len(triplets1)}"
        
        # Test avec format flèches
        text2 = "Alice -> knows -> Bob\nBob -> works_at -> Google"
        triplets2 = extractor._parse_triplets(text2)
        assert len(triplets2) == 2, f"Attendu 2 triplets, obtenu {len(triplets2)}"
        
        print(f"   ✅ Parsing valide")
        print(f"      - Format parenthèses: {len(triplets1)} triplets")
        print(f"      - Format flèches: {len(triplets2)} triplets")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_triplet_validation():
    """Teste la validation des triplets."""
    print("🧪 Test 4: Validation des triplets...")
    try:
        from triplet_extractor import TripletExtractor
        
        # Valide
        assert TripletExtractor.validate_triplet("Alice", "knows", "Bob") == True
        
        # Invalid (vide)
        assert TripletExtractor.validate_triplet("", "knows", "Bob") == False
        
        # Invalid (auto-référence)
        assert TripletExtractor.validate_triplet("Alice", "knows", "Alice") == False
        
        print(f"   ✅ Validation correcte")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_prompt_extractor():
    """Teste l'extraction des prompts."""
    print("🧪 Test 5: Extraction des prompts...")
    try:
        from prompt_extractor import PromptExtractor
        
        # Test avec dict simple
        payload = {"formatted_prompt": "Test prompt"}
        result = PromptExtractor.extract_from_payload(payload)
        assert result == "Test prompt", f"Attendu 'Test prompt', obtenu '{result}'"
        
        # Test avec fallback
        payload = {"other": "data"}
        result = PromptExtractor.extract_from_payload(payload)
        assert isinstance(result, str), "Résultat doit être string"
        
        print(f"   ✅ Extraction valide")
        return True
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def main():
    """Exécute tous les tests."""
    print("\n" + "="*60)
    print("🧪 TESTS DE VALIDATION - REFACTORISATION RAG GRAPH")
    print("="*60)
    
    tests = [
        test_imports,
        test_config,
        test_triplet_parser,
        test_triplet_validation,
        test_prompt_extractor,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Résultat: {passed}/{total} tests réussis")
    print("="*60)
    
    if passed == total:
        print("✅ TOUS LES TESTS RÉUSSIS!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
