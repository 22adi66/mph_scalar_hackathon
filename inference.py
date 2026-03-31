"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.baseline import run_baseline_all_tasks

if __name__ == "__main__":
    api_key = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    if not api_key:
        print("Error: HF_TOKEN or OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    print(f"Running baseline with model {model} at {api_base_url}")
    scores = run_baseline_all_tasks(api_key=api_key, api_base_url=api_base_url, model=model)

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    for task_id in ["easy", "medium", "hard"]:
        s = scores[task_id]
        print(f"  {task_id:8s}: {s['score']:.4f}  ({s['details']})")
    print(f"\n  Aggregate: {scores['aggregate_score']:.4f}")
