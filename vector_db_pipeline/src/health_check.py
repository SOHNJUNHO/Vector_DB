"""
health_check.py — Verify Ollama is reachable and the model is available.
"""

import sys
import time

import requests
from openai import OpenAI


def check_ollama(api_base: str, timeout: int = 10) -> bool:
    """Check Ollama's /api/tags endpoint. Returns True if reachable."""
    tags_url = api_base.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        resp = requests.get(tags_url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def check_vlm_generation(api_base: str, model_name: str) -> bool:
    """Quick smoke test — send a minimal text prompt."""
    try:
        client = OpenAI(api_key="ollama", base_url=api_base)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say 'OK' in one word."}],
            max_tokens=10,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        print(f"[HealthCheck] Generation test passed: '{text}'")
        return True
    except Exception as e:
        print(f"[HealthCheck] Generation test failed: {e}")
        return False


def run_all(
    api_base: str,
    model_name: str,
    retries: int = 3,
    delay: int = 5,
):
    """Run health checks with retries. Exits with code 1 on failure."""
    print("\n=== Ollama Health Check ===")
    print(f"  Endpoint: {api_base}")
    print(f"  Model: {model_name}\n")

    if check_ollama(api_base):
        print("[HealthCheck] Ollama server reachable.")
    else:
        print("[HealthCheck] Ollama not reachable — is 'ollama serve' running?")
        sys.exit(1)

    for attempt in range(1, retries + 1):
        print(f"  Attempt {attempt}/{retries}...")
        if check_vlm_generation(api_base, model_name):
            print("\n[HealthCheck] All checks passed!")
            return
        if attempt < retries:
            print(f"  Retrying in {delay}s...")
            time.sleep(delay)

    print("\n[HealthCheck] Generation test failed after all retries.")
    sys.exit(1)
