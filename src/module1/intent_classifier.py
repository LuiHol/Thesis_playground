
# intent_classifier_llm.py
"""
Simplified LLM-based intent classifier for rugby queries.
- Minimal, constrained prompt to the LLM (JSON-only).
- Strict validation against known intents.
- No edit distance, no typo maps, no partial matches.
- Optional tiny LRU cache.
"""

import logging, time, re, json
from typing import Dict, Optional, Union, List
from collections import OrderedDict
from src.module1.shared.llm_client import LLMClient
from utils.load_prompts import load_prompt

# Logging
logger = logging.getLogger(__name__)


class LRUCache:
    """Tiny LRU cache using OrderedDict."""
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        self.cache.clear()


class IntentClassifierLLM:
    """LLM-based intent classifier (simple & strict)."""

    def __init__(self, model_name: Optional[str] = None):
        # Load minimal config
        cfg = load_prompt("intent_classification") or {}

        # Allowed labels (keep this tight)
        self.valid_intents: List[str] = list(cfg.get("valid_intents", [
            "get_stats",         # generic stat lookup
            "compare",           # compare two entities
            "get_roster",        # team roster
            "get_result",        # final score / result
            "get_player_info",   # "who is player 7?" / player bio
            "top_stat_in_game",  # "who scored the most tries?"
            "get_grades"         # get grade for specific game event
        ]))

        # Model settings
        model_cfg = cfg.get("model_config", {})
        self.model_name = model_name or model_cfg.get("llm_model", "phi3:3.8b")

        # Performance settings
        perf_cfg = cfg.get("performance", {})
        self.enable_caching: bool = perf_cfg.get("enable_caching", True)
        self.max_cache_size: int = perf_cfg.get("max_cache_size", 256)
        self.log_performance: bool = perf_cfg.get("log_performance", True)

        # Client + cache
        try:
            self.client = LLMClient(model=self.model_name)
            logger.info(f"LLM intent classifier initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise RuntimeError(f"Intent classifier initialization failed: {str(e)}")

        self.cache = LRUCache(self.max_cache_size) if self.enable_caching else None
        self.cache_hits = 0
        self.cache_misses = 0

    # Public API
    def classify(self, query: str, model: Optional[str] = None, verbose: bool = False) -> str:
        """
        Classify the intent of a user query.
        Returns one of self.valid_intents, or 'get_stats' as a safe default.
        """
        q = (query or "").strip()
        if not q:
            return "get_stats"

        norm = q.lower()
        t0 = time.time()

        # Cache
        if self.cache:
            cached = self.cache.get(norm)
            if cached:
                self.cache_hits += 1
                if verbose:
                    print(f"CACHE → {cached}")
                return cached
            self.cache_misses += 1

        # LLM
        intent = self._classify_with_llm(q, model=model, verbose=verbose)

        # Cache result
        if self.cache:
            self.cache.put(norm, intent)

        # Perf log
        if verbose and self.log_performance:
            elapsed = time.time() - t0
            print(f"Classification took {elapsed:.3f}s")
            print(f"Cache stats: {self.cache_hits} hits, {self.cache_misses} misses")

        return intent

    # Internals
    def _classify_with_llm(self, query: str, model: Optional[str] = None, verbose: bool = False) -> str:
        """
        Ask the LLM to choose exactly one label from valid_intents.
        Constrained JSON output; strict validation.
        """
        labels = ", ".join(self.valid_intents)
        prompt = (
            "You are an intent classifier for rugby queries.\n"
            "Choose exactly ONE label from this set:\n"
            f"{labels}\n\n"
            'Return THE INTENT ONLY (no markdown, no code fences, no extra text): {"intent":"<one of the labels>"}\n'
            "Classification hints (concise):\n"
            "- Phrases like 'how many', 'number of' => get_stats\n"
            "- 'who scored the most/best/top' => top_stat_in_game (NOT get_stats)\n"
            "- 'compare X with Y' => compare\n"
            "- 'roster', 'lineup', 'squad' => get_roster\n"
            "- 'final score', 'result' => get_result\n"
            "- 'grade', 'grades', 'rating,' => get grades\n\n"
            f"Query: {query}\n"
        )

        try:
            raw = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=model or self.model_name,
                temperature=0.0,
                max_tokens=16,  # small to discourage rambling
            )
            if verbose:
                print(f"LLM raw: {raw!r}")
            return self._parse_llm_json_or_label(raw)
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return "get_stats"

    def _parse_llm_json_or_label(self, raw: str) -> str:
        """
        1) Extract the first {...} block (even if wrapped in ``` fences) and parse "intent".
        2) If no JSON parses, try to pull `"intent":"..."` via regex.
        3) As a last resort, scan for labels mentioned in text, but prefer any label other than
           'get_stats' unless it's the only one.
        """
        import json, re
        txt = (raw or "").strip()

        # A) Extract first JSON object and parse
        m = re.search(r"\{.*?\}", txt, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                intent = str(data.get("intent", "")).strip().lower()
                if intent in self.valid_intents:
                    return intent
            except Exception:
                pass

        # B) Try to pull `"intent":"label"` from raw text
        m = re.search(r'"intent"\s*:\s*"([^"]+)"', txt, flags=re.I)
        if m:
            intent = m.group(1).strip().lower()
            if intent in self.valid_intents:
                return intent

        # C) Fallback: find which valid labels appear in the text
        lower = txt.lower()
        found = [lbl for lbl in self.valid_intents if re.search(rf"\b{re.escape(lbl)}\b", lower)]
        if not found:
            return "get_stats"

        non_default = [lbl for lbl in found if lbl != "get_stats"]
        return non_default[0] if non_default else "get_stats"

    # Utilities
    def clear_cache(self):
        if self.cache:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info("Query cache cleared")

    def get_cache_stats(self) -> Dict[str, Union[bool, int, float]]:
        if not self.enable_caching or self.cache is None:
            return {"enabled": False}
        total = self.cache_hits + self.cache_misses
        return {
            "enabled": True,
            "size": len(self.cache.cache),
            "max_size": self.max_cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": (self.cache_hits / total) if total else 0.0,
        }


# Convenience function
def classify_intent_llm(query: str, model: Optional[str] = None, verbose: bool = False) -> str:
    classifier = IntentClassifierLLM(model_name=model)
    return classifier.classify(query, model=model, verbose=verbose)


# Quick self-test (optional)
if __name__ == "__main__":
    tests = [
        "who scored the most tries in the lyon v bayonne game?",
        "compare the fullback from lyon with the centre from toulon",
        "show me lyon's roster",
        "what was the final score?",
        "who is player 7?",
        "how many tackles did player 15 make?",
        "tell me about player dupont",
        "best tackler?",
        "nonsense output to test fallback",
        "lineup for bayonne?",
        "Compare Lyon wing 11 with Toulon fullback on meters gained",
        "how many turnovers did Dupont make",
        "who had the most tackles?",
        "final score lyon vs bayonne",
        "tell me about player 10 from Toulon",
        "best line breaks?",
        "compare 7 and 8 on tackles",
        "  show LYON squad  ",
        "this is gibberish not rugby",
    ]

    clf = IntentClassifierLLM()
    print(f"Using model: {clf.model_name}")
    print("-" * 50)

    for i, q in enumerate(tests, 1):
        intent = clf.classify(q, verbose=True)
        print(f"{i:02d}. {q}  ->  {intent}")
        print("-" * 30)

    print("\n=== Cache Statistics ===")
    print(clf.get_cache_stats())
