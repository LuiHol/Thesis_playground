# simple_data_handler.py
"""
Simple JSON-based data handler for rugby queries.
Directly reads and queries JSON files without complex database operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured result for data queries."""
    intent: str
    data: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SimpleRugbyDataHandler:
    """Simple rugby data handler using direct JSON operations."""
    
    def __init__(self):
        """Initialize and load JSON data."""
        self.games = []
        self.events = []
        self.players = []
        
        self._load_data()
        logger.info("Simple rugby data handler initialized")
    
    def _load_data(self):
        """Load all JSON data files."""
        try:
            # Load game data
            game_files = list(DATA_DIR.glob("game*.json"))
            for game_file in game_files:
                if game_file.exists():
                    logger.info(f"Loading game file: {game_file}")
                    with open(game_file, 'r') as f:
                        game_data = json.load(f)
                        self.games.append(game_data)
                        
                        # Extract players from rosters
                        self._extract_players_from_game(game_data)
            
            # Load event data
            event_files = list(DATA_DIR.glob("*-events.json"))
            for event_file in event_files:
                if event_file.exists():
                    logger.info(f"Loading event file: {event_file}")
                    with open(event_file, 'r') as f:
                        events_data = json.load(f)
                        self.events.extend(events_data)
            
            logger.info(f"Loaded {len(self.games)} games, {len(self.events)} events, {len(self.players)} players")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _extract_players_from_game(self, game_data):
        """Extract player information from game roster data."""
        game_id = game_data.get('id')
        
        # Home team players
        home_team = game_data.get('home_team', {})
        home_roster = home_team.get('roster', [])
        for player_data in home_roster:
            player_info = {
                'game_id': game_id,
                'team_id': home_team.get('id'),
                'team_name': home_team.get('name'),
                'team_type': 'home',
                'jersey_number': player_data.get('jersey_number'),
                'position': player_data.get('roster_position'),
                'player_id': player_data.get('player', {}).get('id'),
                'first_name': player_data.get('player', {}).get('first_name'),
                'last_name': player_data.get('player', {}).get('last_name'),
                'country': player_data.get('player', {}).get('country'),
                'weight': player_data.get('player', {}).get('weight'),
                'height': player_data.get('player', {}).get('height')
            }
            self.players.append(player_info)
        
        # Away team players
        away_team = game_data.get('away_team', {})
        away_roster = away_team.get('roster', [])
        for player_data in away_roster:
            player_info = {
                'game_id': game_id,
                'team_id': away_team.get('id'),
                'team_name': away_team.get('name'),
                'team_type': 'away',
                'jersey_number': player_data.get('jersey_number'),
                'position': player_data.get('roster_position'),
                'player_id': player_data.get('player', {}).get('id'),
                'first_name': player_data.get('player', {}).get('first_name'),
                'last_name': player_data.get('player', {}).get('last_name'),
                'country': player_data.get('player', {}).get('country'),
                'weight': player_data.get('player', {}).get('weight'),
                'height': player_data.get('player', {}).get('height')
            }
            self.players.append(player_info)
    
    def get_data_for_query(self, entities: Dict[str, List[str]], intent: str) -> QueryResult:
        """
        Main method to handle data queries based on entities and intent.
        
        Args:
            entities: Extracted entities from NER
            intent: Classified intent from LLM
            
        Returns:
            QueryResult with structured data
        """
        try:
            logger.debug(f"Processing query - Intent: {intent}, Entities: {entities}")
            
            if intent == "get_player_info":
                return self._handle_get_player_info(entities)
            elif intent == "get_stats":
                return self._handle_get_stats(entities)
            elif intent == "get_rankings" or intent == "top_stat_in_game":
                return self._handle_get_rankings(entities)
            elif intent == "get_team_info":
                return self._handle_get_team_info(entities)
            elif intent == "compare":
                return self._handle_compare(entities)
            elif intent == "get_game_info":
                return self._handle_get_game_info(entities)
            else:
                return QueryResult(
                    intent=intent,
                    data=[],
                    success=False,
                    error=f"Unknown intent: {intent}"
                )
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return QueryResult(
                intent=intent,
                data=[],
                success=False,
                error=str(e)
            )
    
    def _handle_get_player_info(self, entities: Dict[str, List[str]]) -> QueryResult:
        """Handle player information queries."""
        players = entities.get("players", [])
        teams = entities.get("teams", [])
        
        results = []
        
        for player_ref in players:
            # Find player by jersey number or name
            found_players = []
            
            # Try jersey number first
            if player_ref.isdigit():
                jersey_num = int(player_ref)
                for player in self.players:
                    if player['jersey_number'] == jersey_num:
                        # If team is specified, filter by team
                        if teams:
                            for team_name in teams:
                                if team_name.lower() in player['team_name'].lower():
                                    found_players.append(player)
                        else:
                            found_players.append(player)
            
            # Try by name
            else:
                for player in self.players:
                    full_name = f"{player['first_name']} {player['last_name']}"
                    if (player_ref.lower() in full_name.lower() or 
                        player_ref.lower() in player['last_name'].lower()):
                        found_players.append(player)
            
            # Add found players to results in expected format
            for player in found_players:
                # Format player name properly
                player_name = f"{player['first_name']} {player['last_name']}"
                results.append({
                    "player_id": player['player_id'],
                    "name": player_name,
                    "jersey_number": player['jersey_number'],
                    "position": player['position'],
                    "team": player['team_name'],
                    "team_type": player['team_type'],
                    "game_id": player['game_id']
                })
        
        return QueryResult(
            intent="get_player_info",
            data=results,
            success=len(results) > 0,
            error=None if results else "No matching players found"
        )
    
    def _handle_get_stats(self, entities: Dict[str, List[str]]) -> QueryResult:
        """Handle statistics queries."""
        players = entities.get("players", [])
        teams = entities.get("teams", [])
        event_types = entities.get("event_types", [])
        
        results = []
        
        # Get player stats
        if players:
            for player_ref in players:
                player_info = self._find_player(player_ref, teams)
                if player_info:
                    stats = self._get_player_event_stats(player_info['player_id'], event_types)
                    player_name = f"{player_info['first_name']} {player_info['last_name']}"
                    results.append({
                        "player": {
                            "name": player_name,
                            "jersey_number": player_info['jersey_number'],
                            "position": player_info['position'],
                            "team": player_info['team_name'],
                            "player_id": player_info['player_id']
                        },
                        "statistics": stats,
                        "event_types": event_types
                    })
        
        # Get team stats
        elif teams:
            for team_name in teams:
                team_players = [p for p in self.players if team_name.lower() in p['team_name'].lower()]
                if team_players:
                    team_stats = self._get_team_event_stats(team_players[0]['team_id'], event_types)
                    results.append({
                        "team": team_name,
                        "team_id": team_players[0]['team_id'],
                        "statistics": team_stats,
                        "event_types": event_types
                    })
        
        return QueryResult(
            intent="get_stats",
            data=results,
            success=len(results) > 0,
            error=None if results else "No matching data found"
        )
    
    def _handle_get_rankings(self, entities: Dict[str, List[str]]) -> QueryResult:
        """Handle ranking queries (who scored the most, etc.)."""
        teams = entities.get("teams", [])
        event_types = entities.get("event_types", [])
        
        results = []
        
        # Find relevant events
        relevant_events = []
        for event in self.events:
            # Filter by event type if specified
            if event_types:
                event_type_match = False
                for event_type in event_types:
                    if event_type.lower() in event.get('type', '').lower():
                        event_type_match = True
                        break
                if not event_type_match:
                    continue
            
            # Filter by team if specified
            if teams:
                team_match = False
                event_team_id = event.get('team_id')
                for team_name in teams:
                    team_players = [p for p in self.players if team_name.lower() in p['team_name'].lower()]
                    if team_players and event_team_id == team_players[0]['team_id']:
                        team_match = True
                        break
                if not team_match:
                    continue
            
            relevant_events.append(event)
        
        # Count events and aggregate grades by player
        player_stats = {}
        for event in relevant_events:
            player_id = event.get('player_id')
            if player_id:
                if player_id not in player_stats:
                    player_stats[player_id] = {
                        'count': 0,
                        'grades': [],
                        'grade_sum': 0,
                        'avg_grade': 0
                    }
                
                player_stats[player_id]['count'] += 1
                
                # Add grade if available
                grade = self._extract_grade(event)
                if grade is not None:
                    player_stats[player_id]['grades'].append(grade)
                    player_stats[player_id]['grade_sum'] += grade
                    player_stats[player_id]['avg_grade'] = (
                        player_stats[player_id]['grade_sum'] / len(player_stats[player_id]['grades'])
                    )
        
        # Find top performer (by count, then by average grade)
        if player_stats:
            # Sort by count first, then by average grade
            top_player_id = max(player_stats.keys(), key=lambda pid: (
                player_stats[pid]['count'],
                player_stats[pid]['avg_grade'] if player_stats[pid]['grades'] else 0
            ))
            
            top_stats = player_stats[top_player_id]
            
            # Find player info
            top_player = None
            for player in self.players:
                if player['player_id'] == top_player_id:
                    top_player = player
                    break
            
            if top_player:
                # Format for interface compatibility
                player_name = f"{top_player['first_name']} {top_player['last_name']}"
                results.append({
                    "query_type": "which_team",
                    "top_players": [{
                        "player": {
                            "name": player_name,
                            "jersey_number": top_player['jersey_number'],
                            "position": top_player['position'],
                            "team": top_player['team_name'],
                            "player_id": top_player['player_id']
                        },
                        "count": top_stats['count'],
                        "avg_grade": top_stats['avg_grade'] if top_stats['grades'] else None
                    }],
                    "event_types": event_types,
                    "total_events": len(relevant_events)
                })
        
        return QueryResult(
            intent="get_rankings",
            data=results,
            success=len(results) > 0,
            error=None if results else "No ranking data found"
        )
    
    def _handle_get_team_info(self, entities: Dict[str, List[str]]) -> QueryResult:
        """Handle team information queries."""
        teams = entities.get("teams", [])
        
        results = []
        for team_name in teams:
            team_players = [p for p in self.players if team_name.lower() in p['team_name'].lower()]
            if team_players:
                results.append({
                    "type": "team_info",
                    "team_name": team_players[0]['team_name'],
                    "players": team_players
                })
        
        return QueryResult(
            intent="get_team_info",
            data=results,
            success=len(results) > 0,
            error=None if results else "No matching teams found"
        )
    
    def _handle_compare(self, entities: Dict[str, List[str]]) -> QueryResult:
        """Handle comparison queries."""
        players = entities.get("players", [])
        teams = entities.get("teams", [])
        
        results = []
        
        if len(players) >= 2:
            comparison_data = []
            for player_ref in players:
                player_info = self._find_player(player_ref, teams)
                if player_info:
                    stats = self._get_player_event_stats(player_info['player_id'])
                    comparison_data.append({
                        "player": player_info,
                        "statistics": stats
                    })
            
            if len(comparison_data) >= 2:
                results.append({
                    "type": "player_comparison",
                    "comparison_data": comparison_data
                })
        
        return QueryResult(
            intent="compare",
            data=results,
            success=len(results) > 0,
            error=None if results else "Insufficient data for comparison"
        )
    
    def _extract_grade(self, event: Dict[str, Any]) -> Optional[int]:
        """Extract grade from event metadata."""
        metadata = event.get('metadata', {})
        
        # Try different possible grade locations
        grade = metadata.get('grade')
        if grade is not None:
            try:
                return int(grade)
            except (ValueError, TypeError):
                pass
        
        # Try direct grade field
        grade = event.get('grade')
        if grade is not None:
            try:
                return int(grade)
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _handle_get_game_info(self, entities: Dict[str, List[str]]) -> QueryResult:
        """Handle game information queries."""
        results = []
        
        for game in self.games:
            game_info = {
                "game_id": game.get('id'),
                "round": game.get('round'),
                "start_time": game.get('start_time'),
                "competition": game.get('competition'),
                "venue": game.get('venue'),
                "home_team": game.get('home_team', {}).get('name'),
                "away_team": game.get('away_team', {}).get('name')
            }
            results.append(game_info)
        
        return QueryResult(
            intent="get_game_info",
            data=results,
            success=len(results) > 0
        )
    
    def _find_player(self, player_ref: str, teams: List[str] = None) -> Optional[Dict[str, Any]]:
        """Find a player by reference (jersey number or name)."""
        # Try jersey number first
        if player_ref.isdigit():
            jersey_num = int(player_ref)
            for player in self.players:
                if player['jersey_number'] == jersey_num:
                    # If team is specified, filter by team
                    if teams:
                        for team_name in teams:
                            if team_name.lower() in player['team_name'].lower():
                                return player
                    else:
                        return player
        
        # Try by name
        else:
            for player in self.players:
                full_name = f"{player['first_name']} {player['last_name']}"
                if (player_ref.lower() in full_name.lower() or 
                    player_ref.lower() in player['last_name'].lower()):
                    return player
        
        return None
    
    def _get_player_event_stats(self, player_id: int, event_types: List[str] = None) -> Dict[str, Any]:
        """Get event statistics for a specific player."""
        player_events = [e for e in self.events if e.get('player_id') == player_id]
        
        # Filter by event types if specified
        if event_types:
            filtered_events = []
            for event in player_events:
                for event_type in event_types:
                    if event_type.lower() in event.get('type', '').lower():
                        filtered_events.append(event)
                        break
            player_events = filtered_events
        
        # Count by event type
        event_counts = {}
        grade_stats = {}
        
        for event in player_events:
            event_type = event.get('type', 'unknown')
            
            # Count events
            if event_type not in event_counts:
                event_counts[event_type] = 0
            event_counts[event_type] += 1
            
            # Process grades if available
            grade = self._extract_grade(event)
            if grade is not None:
                if event_type not in grade_stats:
                    grade_stats[event_type] = {
                        'grades': [],
                        'count': 0,
                        'sum': 0,
                        'avg': 0,
                        'min': None,
                        'max': None
                    }
                
                grade_stats[event_type]['grades'].append(grade)
                grade_stats[event_type]['count'] += 1
                grade_stats[event_type]['sum'] += grade
                
                if grade_stats[event_type]['min'] is None or grade < grade_stats[event_type]['min']:
                    grade_stats[event_type]['min'] = grade
                if grade_stats[event_type]['max'] is None or grade > grade_stats[event_type]['max']:
                    grade_stats[event_type]['max'] = grade
                
                grade_stats[event_type]['avg'] = grade_stats[event_type]['sum'] / grade_stats[event_type]['count']
        
        return {
            "total_events": len(player_events),
            "event_counts": event_counts,
            "grade_stats": grade_stats
        }
    
    def _get_team_event_stats(self, team_id: int, event_types: List[str] = None) -> Dict[str, Any]:
        """Get event statistics for a specific team."""
        team_events = [e for e in self.events if e.get('team_id') == team_id]
        
        # Filter by event types if specified
        if event_types:
            filtered_events = []
            for event in team_events:
                for event_type in event_types:
                    if event_type.lower() in event.get('type', '').lower():
                        filtered_events.append(event)
                        break
            team_events = filtered_events
        
        # Count by event type and aggregate grades
        event_counts = {}
        grade_stats = {}
        
        for event in team_events:
            event_type = event.get('type', 'unknown')
            
            # Count events
            if event_type not in event_counts:
                event_counts[event_type] = 0
            event_counts[event_type] += 1
            
            # Process grades if available
            grade = self._extract_grade(event)
            if grade is not None:
                if event_type not in grade_stats:
                    grade_stats[event_type] = {
                        'grades': [],
                        'count': 0,
                        'sum': 0,
                        'avg': 0,
                        'min': None,
                        'max': None
                    }
                
                grade_stats[event_type]['grades'].append(grade)
                grade_stats[event_type]['count'] += 1
                grade_stats[event_type]['sum'] += grade
                
                if grade_stats[event_type]['min'] is None or grade < grade_stats[event_type]['min']:
                    grade_stats[event_type]['min'] = grade
                if grade_stats[event_type]['max'] is None or grade > grade_stats[event_type]['max']:
                    grade_stats[event_type]['max'] = grade
                
                grade_stats[event_type]['avg'] = grade_stats[event_type]['sum'] / grade_stats[event_type]['count']
        
        return {
            "total_events": len(team_events),
            "event_counts": event_counts,
            "grade_stats": grade_stats
        }
