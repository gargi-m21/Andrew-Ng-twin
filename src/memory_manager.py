import json
import os
from langchain_core.messages import HumanMessage, AIMessage
from src.config import LONG_TERM_MEMORY_FILE

class DualLayerMemoryManager:
    def __init__(self):
        self.short_term_store = {}
        # Make sure the data directory exists
        os.makedirs(os.path.dirname(LONG_TERM_MEMORY_FILE), exist_ok=True)
        if not os.path.exists(LONG_TERM_MEMORY_FILE):
            with open(LONG_TERM_MEMORY_FILE, 'w') as f:
                json.dump({}, f)

    def get_short_term(self, session_id: str):
        """Fetches the immediate multi-turn chat history for the current session."""
        if session_id not in self.short_term_store:
            self.short_term_store[session_id] = []
        return self.short_term_store[session_id]

    def add_short_term(self, session_id: str, human_msg: str, ai_msg: str):
        """Appends the latest dialogue exchange to the rolling memory thread."""
        history = self.get_short_term(session_id)
        history.append(HumanMessage(content=human_msg))
        history.append(AIMessage(content=ai_msg))
        
        # Keep rolling frame limit of last 10 messages to optimize latency
        if len(history) > 10:
            self.short_term_store[session_id] = history[-10:]

    def get_long_term_summary(self, session_id: str) -> str:
        """Retrieves persistent user context across sessions."""
        with open(LONG_TERM_MEMORY_FILE, 'r') as f:
            profiles = json.load(f)
        return profiles.get(session_id, "This is your first interaction with this student. Introduce yourself warmly and ask what project they are currently building.")

    def update_long_term_summary(self, session_id: str, new_insights: str):
        """Saves evolving structural data discovered about the student."""
        with open(LONG_TERM_MEMORY_FILE, 'r') as f:
            profiles = json.load(f)
        
        profiles[session_id] = new_insights
        with open(LONG_TERM_MEMORY_FILE, 'w') as f:
            json.dump(profiles, f, indent=4)