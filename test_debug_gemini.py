import json
from server.baseline import run_single_task
import os
import time
import requests

api_key = os.environ.get("OPENAI_API_KEY", "")
api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Hack: inject time.sleep to avoid rate limits when OpenAI client is called
# In baseline.py, run_single_task loops in a for loop. 
# We can't trivially inject sleep without editing baseline.py.
# But wait, we can just replace baseline.py content quickly.
