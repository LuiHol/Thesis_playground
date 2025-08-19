'''------------ OpenwebUI API ------------'''

# llm_client.py
import os
from dotenv import load_dotenv
import requests
from typing import Optional

load_dotenv()


class LLMClient:
    # Class-level default model
    DEFAULT_MODEL = "phi3:3.8b"

    def __init__(
            self,
            model: Optional[str] = None,
            title: str = "RugbyBot-Router",
            referer: str = "http://localhost"
    ):
        self.api_key = os.getenv("OPENWEBUI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENWEBUI_API_KEY environment variable not set")

        # Use provided model or class default
        self.model = model or self.DEFAULT_MODEL
        self.referer = referer
        self.title = title
        self.base_url = "https://gpt.matchsense.dev/api/chat/completions"

    def chat(
            self,
            messages: list[dict],
            temperature: float = 0.0,
            max_tokens: int = 50,
            model: Optional[str] = None  # Allow per-call model override
    ) -> str:

        # Use per-call model, instance model, or class default
        active_model = model or self.model

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.HTTPError as err:
            print(f"[LLMClient] HTTP Error: {err}")
            print(f"[LLMClient] Response Body: {response.text}")
            return "[LLMClient] ERROR"

        except Exception as e:
            print(f"[LLMClient] Unexpected Error: {e}")
            return "[LLMClient] ERROR"

    @classmethod
    def set_default_model(cls, model: str):
        """Set the default model for all new instances"""
        cls.DEFAULT_MODEL = model

    def switch_model(self, model: str):
        """Switch this instance to a different model"""
        self.model = model


