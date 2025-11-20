# ragllm
A short introduction to RAG for LLM

## Description

Ce dépôt contient un cours d'introduction aux techniques de RAG (Retrieval-Augmented Generation) pour les LLM, destiné à un niveau M1.

## Structure du projet

```
ragllm/
├── doc/                          # Documentation et slides
│   └── slides_rag.typ           # Slides Typst (30 diapos)
├── data/                         # Documents markdown pour indexation
│   ├── introduction_rag.md      # Introduction au RAG
│   ├── embeddings.md            # Les embeddings
│   ├── bases_vectorielles.md    # Bases de données vectorielles
│   ├── chunking.md              # Stratégies de découpage
│   └── generation_llm.md        # Génération avec LLM
├── data_big/                     # Documents volumineux (ex: romans)
│   └── chartreuse_de_parme_stendhal.md
├── llm_providers.py             # Configuration modulaire des providers LLM
├── rag_demo.py                  # Démonstration RAG en Python (script)
├── rag_demo.ipynb               # Notebook Jupyter pour Colab
└── requirements.txt             # Dépendances Python
```

## Installation

### Option 1 : Utilisation locale (script Python)

1. Cloner le dépôt :
```bash
git clone https://github.com/phelluy/ragllm.git
cd ragllm
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

### Option 2 : Utilisation sur Google Colab (recommandé pour débutants)

Le notebook `rag_demo.ipynb` est prêt pour Google Colab et ne nécessite aucune installation locale.

**Étapes rapides :**
1. Ouvrir le notebook dans Colab : [Lien à venir]
2. Téléverser vos documents `.md` dans Google Drive
3. Configurer votre clé API (voir section suivante)
4. Exécuter les cellules dans l'ordre

Voir la section **Utilisation sur Google Colab** ci-dessous pour le guide détaillé.

## Utilisation

### Option A : Utilisation sur Google Colab (recommandé)

Le notebook `rag_demo.ipynb` est spécialement conçu pour Google Colab avec intégration Google Drive.

#### 1. Préparation de Google Drive

1. **Créer un dossier pour vos documents** dans Google Drive :
   - Ouvrir [Google Drive](https://drive.google.com)
   - Créer un nouveau dossier, par exemple : `m1_csmi_sgdb/data`
   - Y placer vos fichiers markdown (`.md`) à indexer

2. **Organisation recommandée** :
   ```
   Mon Drive/
   └── m1_csmi_sgdb/
       └── rag_demo.ipynb
       └── data/
           ├── introduction_rag.md
           ├── embeddings.md
           ├── chunking.md
           └── ... (vos autres documents)
   ```

#### 2. Obtenir une clé API Codestral (Mistral AI)

Le notebook utilise par défaut l'API Codestral de Mistral AI (gratuite pour usage modéré).

1. **Créer un compte Mistral AI** :
   - Aller sur [console.mistral.ai](https://console.mistral.ai)
   - Créer un compte ou se connecter

2. **Générer une clé API** :
   - Dans la console, aller dans **API Keys**
   - Cliquer sur **Create new key**
   - Copier la clé générée (format : `xxxxxxxxxxxxxxxxxxxxx`)
   - ⚠️ **Conserver cette clé en sécurité** (elle ne sera affichée qu'une fois)

3. **Vérifier les quotas** :
   - Plan gratuit : limites de requêtes/mois
   - Pour usage intensif : envisager un plan payant

#### 3. Configuration du notebook Colab

1. **Ouvrir le notebook** :
   - Option A : Téléverser `rag_demo.ipynb` sur Google Colab
   - Option B : Ouvrir depuis GitHub (si disponible)

2. **Exécuter la cellule 1** : Installation des dépendances
   ```python
   !pip install -q sentence-transformers torch numpy requests
   ```

3. **Exécuter la cellule 2** : Montage de Google Drive
   - Autoriser l'accès à votre Drive quand demandé
   - Modifier le chemin `DRIVE_DATA_DIR` pour pointer vers votre dossier :
     ```python
     DRIVE_DATA_DIR = '/content/drive/MyDrive/m1_csmi_sgdb/data'  # ← Modifier ici
     ```

4. **Exécuter la cellule 4** : Configuration de la clé API

   **Méthode recommandée - Secrets Colab** (sécurisé) :
   - Cliquer sur l'icône 🔑 **Secrets** dans la barre latérale gauche
   - Cliquer sur **Add new secret**
   - Nom : `CODESTRAL_API_KEY`
   - Valeur : votre clé copiée depuis Mistral AI
   - Activer **Notebook access**
   - Dans la cellule 4, décommenter et adapter :
     ```python
     from google.colab import userdata
     os.environ['CODESTRAL_API_KEY'] = userdata.get('CODESTRAL_API_KEY')
     ```

   **Méthode alternative** (moins sécurisé, pour tests uniquement) :
   ```python
   os.environ['CODESTRAL_API_KEY'] = 'votre_clé_ici'  # ⚠️ Ne pas partager le notebook
   ```

5. **Exécuter les cellules suivantes** dans l'ordre :
   - Cellule 5 : Définition des providers et classe RAG
   - Cellule 6 : Initialisation et chargement des documents
   - Cellules 7-10 : Exemples et tests

#### 4. Utilisation interactive

Une fois initialisé, utiliser la fonction `ask()` :

```python
# Poser une question
ask("Qu'est-ce que le RAG ?")

# Avec plus de sources
ask("Comment fonctionnent les embeddings ?", top_k=10)
```

#### 5. Changer de provider LLM

Le notebook supporte plusieurs providers. Pour changer :

```python
# Dans la cellule 6, modifier :
PROVIDER = "MISTRAL_LARGE"  # ou "IRMA_LLMCODE", "PALGANIA_QWEN3", etc.

# Ou dynamiquement après initialisation :
rag.configure_provider("MISTRAL_LARGE")
```

**Providers disponibles** :
- `MISTRAL_CODESTRAL` (défaut) - Codestral via API Mistral
- `MISTRAL_LARGE` - Mistral Large (nécessite clé API)
- `IRMA_LLMCODE` - Serveur IRMA (pas de clé nécessaire)
- `PALGANIA_QWEN3` - Serveur Palgania
- `LOCAL_QWEN_CODER` - Serveur local (si vous avez un serveur LLM)

#### 6. Sauvegarder les résultats

La cellule 12 montre comment sauvegarder les résultats dans Google Drive :

```python
output_path = '/content/drive/MyDrive/rag_results.txt'
# Le fichier sera automatiquement synchronisé avec votre Drive
```

### Option B : Script Python local (rag_demo.py)

Pour une utilisation en ligne de commande sans notebook.

Pour compiler les slides Typst en PDF :
```bash
typst compile doc/slides_rag.typ
```

### Option B : Script Python local (rag_demo.py)

Pour une utilisation en ligne de commande sans notebook.

#### Configuration des providers

Le système supporte plusieurs providers LLM via le fichier `llm_providers.py`.

**Sélectionner un provider** :
```bash
# Via variable d'environnement
export LLM_PROVIDER=MISTRAL_CODESTRAL
export CODESTRAL_API_KEY=votre_clé_ici

# Puis lancer
python rag_demo.py
```

**Ou modifier directement dans le code** :
```python
# Dans rag_demo.py, ligne ~330
rag = SimpleRAG(data_dir="data_big", provider_name="MISTRAL_CODESTRAL")
```

#### Lancer la démonstration

```bash
python rag_demo.py
```

Le script va :
1. Charger les documents markdown du dossier configuré
2. Créer des embeddings avec le modèle `all-MiniLM-L6-v2`
3. Effectuer des recherches de similarité sur des exemples de questions
4. Proposer un mode interactif pour tester vos propres questions

#### Mode interactif

Après les exemples, le script propose un mode interactif où vous pouvez poser vos propres questions sur le RAG.

### Compiler les slides

Pour compiler les slides Typst en PDF :
```bash
typst compile doc/slides_rag.typ
```

## Contenu du cours

Les slides couvrent :
- Introduction et motivation
- Concepts fondamentaux du RAG
- Architecture d'un système RAG
- Embeddings et similarité
- Stockage vectoriel
- Récupération de documents
- Génération augmentée
- Techniques avancées (HyDE, Re-ranking, Multi-query)
- Métriques d'évaluation (Précision, Rappel, MRR, NDCG)
- Mise en pratique

## Technologies utilisées

- **Sentence Transformers** : Modèles d'embeddings (all-MiniLM-L6-v2)
- **PyTorch** : Backend pour les modèles d'embedding
- **NumPy** : Calculs de similarité cosinus
- **Requests** : Appels aux APIs LLM
- **Typst** : Système de composition des slides
- **Google Colab** : Environnement notebook cloud (optionnel)

## Configuration des providers LLM

Le système utilise une architecture modulaire avec support de multiples providers via `llm_providers.py`.

### Providers disponibles

| Provider | Description | Clé API requise | URL |
|----------|-------------|-----------------|-----|
| **MISTRAL_CODESTRAL** | Codestral (défaut) | `CODESTRAL_API_KEY` | `codestral.mistral.ai` |
| **MISTRAL_LARGE** | Mistral Large | `MISTRAL_API_KEY` | `api.mistral.ai` |
| **IRMA_LLMCODE** | Serveur IRMA | Non | `llmcode.math.unistra.fr:8090` |
| **PALGANIA_QWEN3** | Qwen3-30B | `TEXTSYNTH_API_KEY` | `palgania.ovh:8106` |
| **LOCAL_QWEN_CODER** | Serveur local | Non | `127.0.0.1:8080` |

### Configuration des clés API

**Méthode 1 - Variables d'environnement (recommandé)** :
```bash
export CODESTRAL_API_KEY="votre_clé_ici"
export MISTRAL_API_KEY="votre_clé_ici"
```

**Méthode 2 - Dans le code** :
```python
rag = SimpleRAG(
    data_dir="data",
    provider_name="MISTRAL_CODESTRAL",
    api_key="votre_clé_directement"
)
```

**Méthode 3 - Secrets Colab** (notebook uniquement) :
```python
from google.colab import userdata
os.environ['CODESTRAL_API_KEY'] = userdata.get('CODESTRAL_API_KEY')
```

### Changer de provider

**Au démarrage** :
```python
rag = SimpleRAG(provider_name="MISTRAL_LARGE")
```

**Dynamiquement** :
```python
rag.configure_provider("IRMA_LLMCODE")
rag.configure_provider("PALGANIA_QWEN3", override_model="Qwen3-72B")
```

## Troubleshooting

### Erreur : "Clé API manquante"
- Vérifier que la variable d'environnement est bien définie
- Dans Colab : vérifier que le secret est activé et accessible au notebook

### Erreur : "Timeout" ou "Connection refused"
- Vérifier que le serveur LLM est accessible
- Pour serveurs locaux : vérifier qu'il tourne sur le bon port
- Pour APIs cloud : vérifier votre connexion internet

### Documents non chargés depuis Drive
- Vérifier le chemin `DRIVE_DATA_DIR` (doit pointer vers le bon dossier)
- S'assurer que Drive est bien monté (cellule 2)
- Vérifier que les fichiers ont l'extension `.md`

### Modèle d'embedding trop lent
- Première exécution : téléchargement du modèle (~100 MB)
- Utiliser un GPU dans Colab : Runtime → Change runtime type → GPU

## Ressources supplémentaires

- [Documentation Mistral AI](https://docs.mistral.ai/)
- [Console Mistral AI](https://console.mistral.ai) - Créer clés API
- [Sentence Transformers](https://www.sbert.net/)
- [Guide RAG de LangChain](https://python.langchain.com/docs/use_cases/question_answering/)
- [Google Colab Secrets](https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75)

## Notes

- La première exécution téléchargera le modèle d'embedding (~100 MB)
- Les modèles sont mis en cache automatiquement par HuggingFace
- L'API REST doit être accessible pour la génération de réponses
- Les embeddings sont calculés localement (pas via l'API)

## Licence

Voir le fichier LICENSE
