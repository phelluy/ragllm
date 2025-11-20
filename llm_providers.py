"""Configuration des différents providers LLM compatibles OpenAI.

Chaque provider est défini par :
- url : endpoint /v1/chat/completions compatible OpenAI
- model : nom du modèle à utiliser
- api_key_env : nom de la variable d'environnement contenant la clé (None si pas nécessaire)
"""
from dataclasses import dataclass
from typing import Optional, Dict
import os

@dataclass
class ProviderConfig:
    name: str
    url: str
    model: str
    api_key_env: Optional[str] = None

    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        return os.getenv(self.api_key_env, "")

# Registre des providers connus
PROVIDERS: Dict[str, ProviderConfig] = {
    "MISTRAL_LARGE": ProviderConfig(
        name="MISTRAL_LARGE",
        url="https://api.mistral.ai/v1/chat/completions",
        model="mistral-large-latest",
        api_key_env="MISTRAL_API_KEY",
    ),
    "MISTRAL_CODESTRAL": ProviderConfig(
        name="MISTRAL_CODESTRAL",
        url="https://codestral.mistral.ai/v1/chat/completions",
        model="codestral-latest",
        api_key_env="CODESTRAL_API_KEY",
    ),
    "LOCAL_QWEN_CODER": ProviderConfig(
        name="LOCAL_QWEN_CODER",
        url="http://127.0.0.1:8080/v1/chat/completions",
        model="qwen3-32b-instruct",
        api_key_env=None,
    ),
    "IRMA_LLMCODE": ProviderConfig(
        name="IRMA_LLMCODE",
        url="http://llmcode.math.unistra.fr:8090/v1/chat/completions",
        model="qwen2.5-coder-instruct",
        api_key_env=None,
    ),
    "PALGANIA_QWEN3": ProviderConfig(
        name="PALGANIA_QWEN3",
        url="https://palgania.ovh:8106/v1/chat/completions",
        model="Qwen3-30B",
        api_key_env="TEXTSYNTH_API_KEY",
    ),
}

DEFAULT_PROVIDER = "MISTRAL_CODESTRAL"


def get_provider(name: Optional[str] = None,
                 override_model: Optional[str] = None,
                 override_url: Optional[str] = None,
                 api_key: Optional[str] = None) -> ProviderConfig:
    """Retourne la configuration d'un provider.

    Args:
        name: Nom du provider dans PROVIDERS (None => DEFAULT_PROVIDER)
        override_model: Remplace le nom de modèle si fourni
        override_url: Remplace l'URL si fourni
        api_key: Forcer une clé spécifique (sinon variable d'env)
    """
    if name is None:
        name = DEFAULT_PROVIDER
    if name not in PROVIDERS:
        raise ValueError(f"Provider inconnu: {name}. Providers disponibles: {list(PROVIDERS.keys())}")

    base = PROVIDERS[name]
    model = override_model or base.model
    url = override_url or base.url
    key = api_key if api_key is not None else base.api_key()

    return ProviderConfig(name=name, url=url, model=model, api_key_env=base.api_key_env)

__all__ = ["ProviderConfig", "PROVIDERS", "get_provider", "DEFAULT_PROVIDER"]
