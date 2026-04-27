"""
health_check.py — Verify vLLM server is reachable and responsive.
"""

import sys
import time

import requests
from openai import OpenAI


def check_health(api_base: str, timeout: int = 10) -> bool:
    """Check if the vLLM server responds to /health endpoint."""
    health_url = api_base.rstrip("/").removesuffix("/v1") + "/health"
    try:
        resp = requests.get(health_url, timeout=timeout)
        return resp.status_code == 200
    except Exception as e:
        print(f"[HealthCheck] ❌ vLLM not reachable at {health_url}: {e}")
        return False


def check_models(api_base: str) -> bool:
    """Check if the vLLM server lists available models."""
    try:
        client = OpenAI(api_key="not-needed", base_url=api_base)
        models = client.models.list()
        if models.data:
            model_names = [m.id for m in models.data]
            print(f"[HealthCheck] ✅ Available models: {model_names}")
            return True
        print("[HealthCheck] ❌ No models found")
        return False
    except Exception as e:
        print(f"[HealthCheck] ❌ Failed to list models: {e}")
        return False


def check_vlm_generation(api_base: str, model_name: str) -> bool:
    """Quick smoke test — send a simple text prompt."""
    try:
        client = OpenAI(api_key="not-needed", base_url=api_base)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'OK' in one word."}
            ],
            max_tokens=10,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        print(f"[HealthCheck] ✅ Generation test passed: '{text}'")
        return True
    except Exception as e:
        print(f"[HealthCheck] ❌ Generation test failed: {e}")
        return False


def run_all(api_base: str, model_name: str, retries: int = 3, delay: int = 5):
    """Run all health checks with retries."""
    print("\n=== vLLM Health Check ===")
    print(f"  Endpoint: {api_base}")
    print(f"  Model: {model_name}\n")

    for attempt in range(1, retries + 1):
        print(f"  Attempt {attempt}/{retries}...")

        if not check_health(api_base):
            if attempt < retries:
                print(f"  Retrying in {delay}s...\n")
                time.sleep(delay)
                continue
            print("\n❌ vLLM server is not reachable. Please start it first.")
            sys.exit(1)

        if not check_models(api_base):
            if attempt < retries:
                print(f"  Retrying in {delay}s...\n")
                time.sleep(delay)
                continue
            print("\n❌ Could not retrieve models.")
            sys.exit(1)

        if not check_vlm_generation(api_base, model_name):
            if attempt < retries:
                print(f"  Retrying in {delay}s...\n")
                time.sleep(delay)
                continue
            print("\n❌ Generation test failed.")
            sys.exit(1)

        # All passed
        print("\n✅ All checks passed!")
        return

    print("\n❌ All retries exhausted.")
    sys.exit(1)
