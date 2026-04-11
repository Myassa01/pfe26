"""Client Ollama pour l'inférence LLM locale."""
import json
import time
import logging
from typing import Optional, Generator
import requests

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""Client Hugging Face pour l'inférence LLM locale ou via hub."""
import logging
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen2-7B-Instruct",  # modèle Hugging Face
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.model_name = model
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"Chargement du modèle {self.model_name} sur {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        logger.info("✓ Modèle chargé et prêt.")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Génère une réponse avec Hugging Face."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if system:
            prompt = f"[System]: {system}\n[User]: {prompt}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=tokens,
            temperature=temp,
            do_sample=True,
            top_k=40,
            top_p=0.9,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    def _check_connection(self) -> None:
        """Vérifie la connexion à Ollama et la disponibilité du modèle."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models", [])]
            
            # Vérifie si le modèle exact est disponible
            if self.model not in available:
                # Cherche une correspondance partielle
                matching = [m for m in available if self.model in m]
                if matching:
                    logger.info(f"Modèle trouvé: {matching[0]}")
                    self.model = matching[0]  # Utilise le nom exact
                else:
                    logger.warning(f"\n[ATTENTION] Modèle '{self.model}' non trouvé.")
                    logger.warning(f"  Modèles disponibles: {available or 'aucun'}")
                    logger.warning(f"  Téléchargez-le avec: ollama pull {self.model}\n")
                    
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Impossible de se connecter à Ollama ({self.base_url}).\n"
                f"  1. Installez Ollama: https://ollama.com\n"
                f"  2. Démarrez-le: ollama serve\n"
                f"  3. Téléchargez un modèle: ollama pull {self.model}"
            )
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de connexion: {e}")

    def _warm_up(self) -> None:
        """Préchauffe le modèle pour réduire le temps de la première requête."""
        try:
            logger.info(f"Préchauffage du modèle {self.model}...")
            start = time.time()
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "warm up",
                    "stream": False,
                    "options": {"num_predict": 5}
                },
                timeout=30
            )
            if r.status_code == 200:
                elapsed = time.time() - start
                logger.info(f"✓ Modèle prêt ({elapsed:.2f}s)")
        except Exception as e:
            logger.warning(f"⚠️ Préchauffage optionnel: {e}")

    def _optimize_prompt(self, prompt: str, max_length: int = 6000) -> str:
        """Optimise le prompt pour les modèles légers."""
        if len(prompt) > max_length:
            # Garde le début et la fin du prompt
            prompt = prompt[:max_length] + "\n...\n[Réponse courte et concise]"
        return prompt

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Génère une réponse complète (non-streaming).
        
        Args:
            prompt: Le prompt utilisateur
            system: Système prompt optionnel
            temperature: Température (par défaut: 0.1)
            max_tokens: Nombre maximum de tokens (par défaut: 200)
        
        Returns:
            La réponse générée
        """
        # Optimisation du prompt
        prompt = self._optimize_prompt(prompt)
        
        # Utilise les valeurs par défaut ou celles passées en paramètre
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Construction du payload
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
                "top_k": 40,  # Réduit pour accélérer
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        
        if system:
            payload["system"] = system
        
        logger.info(f"Génération avec {self.model} (max_tokens={tokens}, temp={temp})")
        start_time = time.time()
        
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            
            elapsed = time.time() - start_time
            response = r.json()["response"]
            logger.info(f"✓ Réponse générée en {elapsed:.2f}s ({len(response)} caractères)")
            
            return response
            
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            raise TimeoutError(
                f"⏱️ Timeout après {elapsed:.2f}s. "
                f"Le modèle {self.model} a mis trop de temps à répondre. "
                f"Essayez de réduire max_tokens ou d'utiliser un modèle plus petit."
            )
        except requests.exceptions.RequestException as e:
            raise Exception(f"❌ Erreur lors de la génération: {e}")

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Génère une réponse en streaming (token par token).
        
        Args:
            prompt: Le prompt utilisateur
            system: Système prompt optionnel
            temperature: Température (par défaut: 0.1)
            max_tokens: Nombre maximum de tokens (par défaut: 200)
        
        Yields:
            Tokens un par un
        """
        # Optimisation du prompt
        prompt = self._optimize_prompt(prompt)
        
        # Utilise les valeurs par défaut ou celles passées en paramètre
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temp,
                "num_predict": tokens,
                "top_k": 40,
                "top_p": 0.9,
            },
        }
        
        if system:
            payload["system"] = system
        
        logger.info(f"Génération en streaming avec {self.model}")
        start_time = time.time()
        token_count = 0
        
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            token_count += 1
                            yield token
                        if data.get("done"):
                            elapsed = time.time() - start_time
                            logger.info(
                                f"✓ Streaming terminé: {token_count} tokens "
                                f"en {elapsed:.2f}s ({token_count/elapsed:.1f} tok/s)"
                            )
                            break
        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            raise TimeoutError(
                f"⏱️ Timeout streaming après {elapsed:.2f}s"
            )
        except Exception as e:
            raise Exception(f"❌ Erreur lors du streaming: {e}")

    def get_model_info(self) -> dict:
        """Récupère les informations du modèle."""
        try:
            r = requests.post(
                f"{self.base_url}/api/show",
                json={"model": self.model},
                timeout=5
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Impossible de récupérer les infos du modèle: {e}")
            return {}