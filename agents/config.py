"""
Configuration module for the Multi-Agent Analyst system.
Contains LLM setup, timezone configuration, and utility functions.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# ---------------------------
# Timezone Configuration
# ---------------------------
NY_TZ = ZoneInfo("America/New_York")


def today_ny_str() -> str:
    """Returns today's date in America/New_York timezone as ISO format string."""
    return datetime.now(NY_TZ).date().isoformat()  # e.g. "2026-01-29"


# ---------------------------
# LLM Configuration
# ---------------------------
def get_llm(model: str = "gpt-4o", temperature: float = 0) -> ChatOpenAI:
    """Returns a configured ChatOpenAI instance."""
    return ChatOpenAI(model=model, temperature=temperature)


# Default LLM instance
llm = get_llm()
