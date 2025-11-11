# Génération avec LLM

La dernière étape du RAG consiste à utiliser un modèle de langage pour générer une réponse basée sur les documents récupérés.

## Construction du prompt

Le prompt doit inclure :
1. Les documents récupérés (contexte)
2. La question de l'utilisateur
3. Des instructions pour le LLM

### Exemple de template

```
Tu es un assistant qui répond aux questions en te basant uniquement sur le contexte fourni.

Contexte:
{documents}

Question: {question}

Instructions:
- Base ta réponse uniquement sur le contexte fourni
- Si l'information n'est pas dans le contexte, dis que tu ne sais pas
- Cite tes sources quand c'est possible

Réponse:
```

## Choix du LLM

### Modèles cloud
- GPT-4, GPT-3.5 (OpenAI)
- Claude (Anthropic)
- Gemini (Google)

**Avantages :** Qualité, pas de setup
**Inconvénients :** Coût, confidentialité

### Modèles open-source
- Mistral 7B
- LLaMA 2/3
- Phi-3
- Gemma

**Avantages :** Gratuit, privé, customisable
**Inconvénients :** Nécessite des ressources

## Paramètres de génération

### Temperature
Contrôle la créativité/randomness
- 0 : Déterministe
- 1 : Plus créatif
- Pour RAG : généralement 0-0.3

### Top-p (nucleus sampling)
Alternative à la temperature
- Considère les tokens dont la probabilité cumulée atteint p
- Valeurs typiques : 0.9-0.95

### Max tokens
Limite la longueur de la réponse
- Dépend du cas d'usage
- Attention au coût

## Optimisation

### Formatting des documents
- Numéroter les sources
- Ajouter des titres
- Structurer clairement

### Instructions claires
- Spécifier le format de réponse attendu
- Donner des exemples
- Préciser les contraintes

### Post-processing
- Extraction des citations
- Vérification de cohérence
- Formatting de la sortie

## Évaluation

Métriques à considérer :
- **Fidélité** : La réponse est-elle basée sur le contexte ?
- **Pertinence** : La réponse répond-elle à la question ?
- **Cohérence** : La réponse est-elle bien structurée ?
- **Complétude** : Tous les aspects sont-ils couverts ?
