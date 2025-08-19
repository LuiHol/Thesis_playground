# entity_extractor_spacy.py
"""
Simplified SpaCy-based Entity Extractor for Rugby queries.
Let spaCy do the heavy lifting, we just categorize the results.
"""

import spacy
from spacy.matcher import Matcher, PhraseMatcher
import re
from typing import Dict, List, Optional
import yaml
import json
from pathlib import Path
import sys


class SpacyEntityExtractor:
    """Extract rugby entities using spaCy."""

    def __init__(self, model_name: str = "en_core_web_md"):
        """Initialize with spaCy model and load configurations."""
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

        # Initialize matchers for specific patterns
        self.matcher = Matcher(self.nlp.vocab)

        # Load configurations
        self._load_yaml_configs()
        self._setup_basic_patterns()

    def _load_yaml_configs(self):
        """Load YAML configuration files."""
        # Get path to shared directory
        project_root = Path(__file__).resolve().parents[3]  # Go up to Thesis/
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        try:
            from config import SHARED_DIR
        except ImportError:
            SHARED_DIR = Path(__file__).parent.parent / "shared"

        # Load position terms
        self.position_terms = set()
        try:
            with open(SHARED_DIR / "jersey_mappings.yaml", "r") as f:
                jersey_data = yaml.safe_load(f)
            if jersey_data and "aliases" in jersey_data:
                for position, variations in jersey_data["aliases"].items():
                    self.position_terms.add(position.lower())
                    if isinstance(variations, list):
                        self.position_terms.update([v.lower() for v in variations])
        except:
            pass

        # Load event terms
        self.event_terms = set()
        try:
            with open(SHARED_DIR / "event_mappings.yaml", "r") as f:
                event_data = yaml.safe_load(f)
            if event_data and "event_mappings" in event_data:
                for event, variations in event_data["event_mappings"].items():
                    self.event_terms.add(event.lower())
                    if isinstance(variations, list):
                        self.event_terms.update([v.lower() for v in variations])
        except:
            pass

    def _setup_basic_patterns(self):
        """Set up only the most essential patterns."""
        # Pattern for "player X" format
        player_pattern = [
            {"LOWER": "player"},
            {"IS_DIGIT": True}
        ]
        self.matcher.add("PLAYER_NUMBER", [player_pattern])

        # Time patterns
        time_patterns = [
            [{"LOWER": "round"}, {"IS_DIGIT": True}],
            [{"LOWER": "match"}, {"IS_DIGIT": True}],
            [{"LOWER": "game"}, {"IS_DIGIT": True}],
            [{"LOWER": "this"}, {"LOWER": "season"}],
            [{"LOWER": "last"}, {"LOWER": {"IN": ["match", "game", "season"]}}]
        ]
        self.matcher.add("TIME_REF", time_patterns)

    def extract(self, query: str, verbose: bool = False) -> Dict[str, List[str]]:
        """
        Extract entities from a rugby query.
        Let spaCy do the work, we just categorize.
        """
        if not query or not query.strip():
            return self._get_empty_structure()

        # Process with spaCy
        doc = self.nlp(query)

        # Initialize result
        entities = self._get_empty_structure()

        # Extract players (numbers and person names)
        entities["players"] = self._extract_players(doc)

        # Extract teams (organizations and locations)
        entities["teams"] = self._extract_teams(doc)

        # Extract positions (from our known list)
        entities["positions"] = self._extract_positions(doc)

        # Extract events (from our known list)
        entities["event_types"] = self._extract_events(doc)

        # Extract time references
        entities["time_reference"] = self._extract_time_references(doc)

        # Remove duplicates
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        if verbose:
            print(f"Query: {query}")
            print(f"SpaCy entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
            print(f"Extracted: {entities}")

        return entities

    def _extract_players(self, doc) -> List[str]:
        """Extract player references."""
        players = []

        # Get jersey numbers from "player X" patterns
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "PLAYER_NUMBER":
                text = doc[start:end].text
                num = re.search(r'\d+', text)
                if num:
                    players.append(num.group())

        # Get standalone jersey numbers with context
        for token in doc:
            if token.text.isdigit() and 1 <= int(token.text) <= 23:
                # Check if preceded by player-related words or followed by 's
                if token.i > 0 and doc[token.i - 1].text.lower() in ["player", "number", "no"]:
                    players.append(token.text)
                elif token.i < len(doc) - 1 and doc[token.i + 1].text == "'s":
                    players.append(token.text)

        # Get person names from NER
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                if name.endswith("'s"):
                    name = name[:-2]
                # Simple check: if it contains a known team name, skip it
                if not any(word in ["Lyon", "Toulon", "Racing", "Bayonne"] for word in ent.text.split()):
                    players.append(name)

        return players

    def _extract_teams(self, doc) -> List[str]:
        """Extract team names."""
        teams = []

        # Get organizations and locations from NER
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "LOC"]:
                name = ent.text
                if name.endswith("'s"):
                    name = name[:-2]
                teams.append(name)

        # Look for capitalized words after team indicators
        indicators = ["from", "for", "against", "versus", "vs", "in", "at"]
        for i, token in enumerate(doc):
            if token.text.lower() in indicators and i < len(doc) - 1:
                next_token = doc[i + 1]
                if next_token.text[0].isupper() and next_token.pos_ == "PROPN":
                    teams.append(next_token.text)

        return teams

    def _extract_positions(self, doc) -> List[str]:
        """Extract position mentions."""
        positions = []

        # Simple token matching against known positions
        for token in doc:
            if token.text.lower() in self.position_terms:
                positions.append(token.text)
            # Check for multi-word positions
            if token.i < len(doc) - 1:
                two_word = f"{token.text} {doc[token.i + 1].text}".lower()
                if two_word in self.position_terms:
                    positions.append(f"{token.text} {doc[token.i + 1].text}")

        return positions

    def _extract_events(self, doc) -> List[str]:
        """Extract event types."""
        events = []

        # Simple token matching against known events
        for token in doc:
            if token.text.lower() in self.event_terms:
                events.append(token.text)
            # Check for multi-word events like "yellow card"
            if token.i < len(doc) - 1:
                two_word = f"{token.text} {doc[token.i + 1].text}".lower()
                if two_word in self.event_terms:
                    events.append(f"{token.text} {doc[token.i + 1].text}")

        return events

    def _extract_time_references(self, doc) -> List[str]:
        """Extract time references."""
        time_refs = []

        # Get pattern matches
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "TIME_REF":
                time_refs.append(doc[start:end].text)

        return time_refs

    def _get_empty_structure(self) -> Dict[str, List[str]]:
        """Return empty entity structure."""
        return {
            "players": [],
            "teams": [],
            "positions": [],
            "event_types": [],
            "time_reference": []
        }


def extract_entities(
        query: str,
        model: Optional[str] = None,
        verbose: bool = False
) -> Dict[str, List[str]]:
    """
    Extract entities from a query using spaCy.
    """
    extractor = SpacyEntityExtractor(model_name=model or "en_core_web_md")
    return extractor.extract(query, verbose=verbose)


if __name__ == "__main__":
    # Test the entity extractor
    test_queries = [
        "How many tackles did player 15 make in round 3?",
        "Compare the fullback from Lyon with the inside centre from Toulon",
        "Which team had the most offloads in the last match?",
        "Show me player 7's tries and tackles this season",
        "Did the wing from Racing score more tries than player 11?",
        "What's the total number of yellow cards for props this season?",
        "Player 9 versus player 10 performance in match 5",
        "Who is the best tackler in Lyon?",
        "Compare Lyon and Bayonne in terms of possession",
        "How did Toulouse perform against Clermont and La Rochelle this season?",
        "Show defensive stats for Racing, Stade Français and Toulon",

        # Specific player names (testing PERSON entity)
        "Did Antoine Dupont score more tries than Romain Ntamack?",
        "Compare Damian Penaud's metres gained with player 14 from Lyon",

        # Complex position queries
        "Which props and hookers got the most turnovers?",
        "Show all red cards for second row players in round 5",

        # Multiple time references
        "Compare round 1 and round 2 performance for Bayonne",
        "How many penalties in the first half versus second half?",

        # Possession and territory metrics
        "Which team had better possession and territory in match 3?",
        "Show possession stats for all matches this season",

        # Mixed entities
        "Did the Toulouse scrum half get more yellow cards than the Lyon fly half?",
        "Player 8 and player 9 from Montpellier versus Pau flankers",

        # Edge cases with numbers
        "Show stats for players 1 through 5 from Bordeaux",
        "How many tries in matches 10, 11, and 12?",

        # Informal queries
        "Lyon's 15 vs Toulon's fullback tackles",
        "Best kicker between Racing and Stade?",

        # Complex event combinations
        "Turnovers leading to tries for Castres",
        "Penalties conceded at scrums and lineouts by Brive",

        # Ambiguous team names
        "How did Racing 92 perform against Racing Metro?",
        "La Rochelle and Stade Rochelais possession comparison"
    ]

    print("=== Testing Simplified SpaCy Entity Extractor ===\n")
    extractor = SpacyEntityExtractor()

    for query in test_queries:
        print(f"Query: {query}")
        entities = extractor.extract(query, verbose=False)
        print(f"Entities: {json.dumps(entities, indent=2)}")
        print("-" * 50)
