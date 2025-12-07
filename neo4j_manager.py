"""
Module pour la gestion de la connexion et l'interaction avec Neo4j.
"""

import logging
from typing import Optional
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from config import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

logger = logging.getLogger(__name__)


class Neo4jManager:
    """
    Gère la connexion et les opérations sur une base Neo4j.
    """

    def __init__(self, url: str = NEO4J_URL, user: str = NEO4J_USER, 
                 password: str = NEO4J_PASSWORD, database: str = NEO4J_DATABASE):
        """
        Initialise le gestionnaire Neo4j.
        
        Args:
            url: URL de connexion Neo4j (ex: bolt://localhost:7687)
            user: Nom d'utilisateur
            password: Mot de passe
            database: Nom de la base de données
        """
        self.url = url
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

    def connect(self) -> bool:
        """
        Établit la connexion à Neo4j.
        
        Returns:
            True si la connexion est réussie, False sinon
        """
        try:
            logger.info(f"🕸️ Connexion à Neo4j ({self.url})...")
            self.driver = GraphDatabase.driver(
                self.url, 
                auth=(self.user, self.password),
                connection_timeout=5.0
            )
            # Test de connexion
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Connexion Neo4j établie.")
            return True
        except ServiceUnavailable as e:
            logger.error(f"❌ Service Neo4j indisponible: {e}")
            return False
        except Neo4jError as e:
            logger.error(f"❌ Erreur Neo4j: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur connexion Neo4j: {e}")
            return False

    def is_connected(self) -> bool:
        """Vérifie si la connexion est active."""
        return self.driver is not None

    def clear_database(self) -> bool:
        """
        Vide la base de données (supprime tous les nœuds et relations).
        
        Returns:
            True si succès, False sinon
        """
        if not self.is_connected():
            logger.warning("Pas de connexion Neo4j, impossible de vider la base")
            return False

        try:
            logger.info("🧹 Nettoyage de la base Neo4j...")
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) as deleted")
                record = result.single()
                count = record["deleted"] if record else 0
            logger.info(f"   → {count} nœuds supprimés.")
            return True
        except Exception as e:
            logger.error(f"⚠️ Erreur lors du nettoyage de Neo4j: {e}")
            return False

    def count_nodes(self) -> int:
        """
        Compte le nombre de nœuds dans la base.
        
        Returns:
            Nombre de nœuds, ou -1 en cas d'erreur
        """
        if not self.is_connected():
            return -1

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                record = result.single()
                return record["count"] if record else 0
        except Exception as e:
            logger.warning(f"Impossible de compter les nœuds: {e}")
            return -1

    def graph_exists(self) -> bool:
        """
        Vérifie si un graphe existe (au moins 1 nœud).
        
        Returns:
            True si le graphe existe, False sinon
        """
        count = self.count_nodes()
        return count > 0

    def close(self):
        """Ferme la connexion."""
        if self.driver:
            try:
                self.driver.close()
                logger.debug("Connexion Neo4j fermée.")
            except Exception as e:
                logger.warning(f"Erreur lors de la fermeture: {e}")
            finally:
                self.driver = None

    def __enter__(self):
        """Support du context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support du context manager."""
        self.close()
