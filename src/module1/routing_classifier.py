import logging

from src.module1.shared.llm_client import LLMClient
from utils.load_prompts import load_prompt
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Single client instance
client = LLMClient()


def classify_query(
        query: str,
        model: Optional[str] = None,
        verbose: bool = False
) -> str:
    try:
        prompt_data: Dict = load_prompt("routing_classifier")
        system_prompt: str = prompt_data.get("system", "")
        examples: List[Dict] = prompt_data.get("examples", [])

        messages = [{"role": "system", "content": system_prompt}]

        for example in examples:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["route"]})

        messages.append({"role": "user", "content": query})

        # Use per-call model override or client's default
        raw = client.chat(
            messages,
            model=model,
            max_tokens=4,
            temperature=0.0
        )

        # ---- Robust normalization ----
        response = (raw or "").strip().lower()

        # Take first token; strip punctuation
        first = response.split()[0] if response else ""
        first = first.strip(".,:;!?'\"-_/\\()[]{}")

        # Map common variants
        if first in {"simple", "s"}:
            normalized_response = "simple"
        elif first in {"complex", "c"}:
            normalized_response = "complex"
        elif first.isdigit():
            # Some models emit numerals; treat as simple by default
            normalized_response = "simple"
        elif "error" in response or "llmclient" in response:
            normalized_response = "simple"
        else:
            normalized_response = first

        if verbose:
            active_model = model or getattr(client, "model", "unknown")
            print(f"Using model: {active_model}")

        if normalized_response not in {"simple", "complex"}:
            logger.warning(
                f"[routing_classifier] Unexpected response: '{raw}' -> '{normalized_response}'. Defaulting to 'simple'."
            )
            return "simple"

        return normalized_response

    except Exception as error:
        logger.error(f"Routing classification error: {error}")
        return "simple"


if __name__ == "__main__":
    test_queries = [
        "How many tackles did player X make in round 3?",
        "Compare player Y's stats with player Z.",
        "What was the final score of match 6?",
        "Has team A improved throughout the season?",
        "Which team had the most offloads?",
        "Was player B more efficient than player C in the last 5 games?",
        "How many yellow cards did team D get?",
        "How consistent was team E across the tournament?",
        "What jersey number does player F wear?",
        "Which team dominated possession in round 2?"
    ]

    for query in test_queries:
        try:
            result = classify_query(query, verbose=False)
            print(f"Query: {query}\n Routing: {result}\n{'-'*50}")
        except Exception as e:
            print(f" Failed to classify: {query}\nReason: {e}")
