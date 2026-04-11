"""Client Hugging Face pour l'inférence LLM locale."""
import time
import logging
from typing import Optional, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HFClient:
    def __init__(
        self,
        model: str = "Qwen/Qwen2-7B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.7,
        max_tokens: int = 512,
        warm_up: bool = True,
    ):
        self.model_name  = model
        self.device      = device
        self.temperature = temperature
        self.max_tokens  = max_tokens

        self._check_connection()
        if warm_up:
            self._warm_up()

    def _check_connection(self) -> None:
        """Charge le modèle et vérifie qu'il est bien disponible."""
        try:
            logger.info(f"Chargement du modèle {self.model_name} sur {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            logger.info("✓ Modèle chargé et prêt.")
        except OSError:
            raise ConnectionError(
                f"Impossible de charger le modèle '{self.model_name}'.\n"
                f"  1. Vérifiez le nom du modèle sur https://huggingface.co/models\n"
                f"  2. Vérifiez votre connexion internet\n"
                f"  3. Authentifiez-vous si nécessaire: huggingface-cli login"
            )
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise

    def _warm_up(self) -> None:
        """Préchauffe le modèle pour réduire le temps de la première requête."""
        try:
            logger.info(f"Préchauffage du modèle {self.model_name}...")
            start = time.time()
            inputs = self.tokenizer("warm up", return_tensors="pt").to(self.device)
            self.model.generate(**inputs, max_new_tokens=5)
            logger.info(f"✓ Modèle prêt ({time.time() - start:.2f}s)")
        except Exception as e:
            logger.warning(f"⚠️ Préchauffage optionnel échoué: {e}")

    def _optimize_prompt(self, prompt: str, max_length: int = 6000) -> str:
        """Tronque les prompts trop longs."""
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "\n...\n[Réponse courte et concise]"
        return prompt

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Génère une réponse complète (non-streaming)."""
        prompt = self._optimize_prompt(prompt)
        temp   = temperature if temperature is not None else self.temperature
        tokens = max_tokens  if max_tokens  is not None else self.max_tokens

        if system:
            prompt = f"[System]: {system}\n[User]: {prompt}"

        logger.info(f"Génération avec {self.model_name} (max_tokens={tokens}, temp={temp})")
        start = time.time()

        try:
            inputs  = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            do_sample = temp > 0

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                temperature=temp if do_sample else None,  # ignoré si greedy
                do_sample=do_sample,
                top_k=40 if do_sample else None,
                top_p=0.9 if do_sample else None,
                repetition_penalty=1.1,
            )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"✓ Réponse en {time.time()-start:.2f}s ({len(response)} chars)")
            return response

        except torch.cuda.OutOfMemoryError:
            raise MemoryError(
                f"⏱️ VRAM insuffisante pour {self.model_name}. "
                f"Réduisez max_tokens ou utilisez un modèle plus petit."
            )
        except Exception as e:
            raise Exception(f"❌ Erreur lors de la génération: {e}")

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Génère une réponse en streaming (token par token)."""
        prompt = self._optimize_prompt(prompt)
        temp   = temperature if temperature is not None else self.temperature
        tokens = max_tokens  if max_tokens  is not None else self.max_tokens

        if system:
            prompt = f"[System]: {system}\n[User]: {prompt}"

        logger.info(f"Streaming avec {self.model_name}")
        start, token_count = time.time(), 0
        do_sample = temp > 0
        try:
            inputs   = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=tokens,
                temperature=temp if do_sample else None,
                do_sample=do_sample,
                top_k=40 if do_sample else None,
                top_p=0.9 if do_sample else None,
                streamer=streamer,
            )
            # Génération dans un thread séparé pour ne pas bloquer le générateur
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            for token in streamer:
                if token:
                    token_count += 1
                    yield token

            thread.join()
            elapsed = time.time() - start
            logger.info(
                f"✓ Streaming terminé: {token_count} tokens "
                f"en {elapsed:.2f}s ({token_count/elapsed:.1f} tok/s)"
            )

        except torch.cuda.OutOfMemoryError:
            raise MemoryError(f"⏱️ VRAM insuffisante pour le streaming.")
        except Exception as e:
            raise Exception(f"❌ Erreur lors du streaming: {e}")

    def get_model_info(self) -> dict:
        """Retourne les informations du modèle chargé."""
        try:
            config = self.model.config.to_dict()
            return {
                "model_name":  self.model_name,
                "device":      self.device,
                "dtype":       str(next(self.model.parameters()).dtype),
                "vocab_size":  config.get("vocab_size"),
                "num_layers":  config.get("num_hidden_layers"),
                "hidden_size": config.get("hidden_size"),
            }
        except Exception as e:
            logger.error(f"Impossible de récupérer les infos du modèle: {e}")
            return {}