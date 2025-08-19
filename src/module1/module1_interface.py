# chat_interface.py
"""
A minimal interactive CLI chat for testing Module 1 (NLU) with LLM-enhanced responses.
"""
import sys
from src.module1.components.module1_orchestrator import Module1Orchestrator
from src.module1.components.routing_classifier import classify_query
from src.module1.shared.llm_client import LLMClient
from utils.load_prompts import load_prompt

# Load prompt from YAML file
CHAT_RESPONSE_PROMPT_TEMPLATE = load_prompt("chat_response")['template']

# Instantiate LLM client
llm = LLMClient()
orchestrator = Module1Orchestrator(verbose=False)

def _sanitize_user_input(text:str) -> str:
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.strip().startswith(("INFO:", "DEBUG:", "WARNING:", "ERROR:"))]
    if not lines:
        return ""
    return lines[-1].strip()

def format_response_with_llm(data: dict, query: str) -> str:
    records = data.get("data", [])
    if not records:
        return "Sorry, I couldn’t find any results for that."

    if isinstance(records, list) and records:
        record = records[0]  # Assume first record is most relevant
    elif isinstance(records, dict):
        record = records
    else:
        record = {"data": records}

    if isinstance(record, dict) and "query_type" in record:
        if record["query_type"] == "which_team" and "winner" in record:
            team_info = record["winner"]
            return f"{team_info['team']} had the most with {team_info['count']}."
        elif record["query_type"] == "which_team" and "top_players" in record:
            if record["top_players"]:
                player = record["top_players"][0]["player"]
                count = record["top_players"][0]["count"]
                return f"{player['team']} (#{player['jersey_number']}) had the most with {count}."

    prompt = CHAT_RESPONSE_PROMPT_TEMPLATE.format(
        query=query.strip(),
        record=str(record)
    )

    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.strip()
    except Exception as e:
        return f"Error formatting response with LLM: {e}"

def run_chat():
    print("Rugby Chat (Module 1 Proof-of-Concept)")
    print("Type 'exit' to quit.\n")

    while True:
        raw_input_text = input("You: ")
        user_input = _sanitize_user_input(raw_input_text)
        if not user_input:
            print("I didnt catch that. Please try again.")
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            orchestrator.close()  # Clean up resources
            break

        try:
            print(f"DEBUG: Testing routing classifier...")
            route = classify_query(user_input, verbose=True)
            print(f"DEBUG: Route = {route}")

            if route == "complex":
                print("Bot: That query is a bit complex - Thinking harder...")
                continue

            # DEBUG: Test orchestrator step by step
            print(f"DEBUG: Testing orchestrator...")

            # Test entity extraction
            print(f"DEBUG: Testing entity extraction...")
            entities = orchestrator.entity_extractor.extract(user_input, verbose=True)
            print(f"DEBUG: Entities = {entities}")

            # Test intent classification
            print(f"DEBUG: Testing intent classification...")
            intent = orchestrator.intent_classifier.classify(user_input, verbose=True)
            print(f"DEBUG: Intent = {intent}")

            # Full orchestrator call
            result = orchestrator.process_query(user_input)

            result = orchestrator.process_query(user_input)

            if not result.success:
                print("Bot: Sorry, I couldn't understand that.")
                if result.error:
                    print(f"Debug: {result.error}")
                continue

            response = format_response_with_llm(result.data, user_input)
            print(f"Bot: {response}")
            
        except Exception as e:
            print(f"Bot: Sorry, something went wrong: {str(e)}")
            continue


if __name__ == "__main__":
    run_chat()
