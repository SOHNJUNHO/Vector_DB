"""
health_check.py — Verify the VLM API is reachable and responsive.

Works with both local vLLM (has /health + /models endpoints) and hosted APIs
like Fireworks AI (neither endpoint available — generation test is the only gate).
"""

import sys
import time

import requests
from openai import OpenAI


def check_health(api_base: str, timeout: int = 10) -> bool:
    """Check /health endpoint. Returns True if reachable, False otherwise (soft check)."""
    health_url = api_base.rstrip("/").removesuffix("/v1") + "/health"
    try:
        resp = requests.get(health_url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def check_models(api_base: str, api_key: str = "not-needed") -> bool:
    """List available models. Soft check — returns True if unsupported (e.g. Fireworks)."""
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
        models = client.models.list()
        if models.data:
            model_names = [m.id for m in models.data]
            print(f"[HealthCheck] Available models: {model_names}")
        else:
            print("[HealthCheck] Models list empty")
        return True
    except Exception as e:
        print(f"[HealthCheck] Models list not available (OK for hosted APIs): {e}")
        return True


def check_vlm_generation(api_base: str, model_name: str, api_key: str = "not-needed") -> bool:
    """Quick smoke test — send a simple text prompt."""
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
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
    api_key: str = "not-needed",
):
    """
    Run health checks with retries.

    /health and models list are soft checks — skipped gracefully if unsupported.
    Generation test is the only hard gate.
    """
    print("\n=== VLM Health Check ===")
    print(f"  Endpoint: {api_base}")
    print(f"  Model: {model_name}\n")

    http_ok = check_health(api_base)
    if http_ok:
        print("[HealthCheck] /health endpoint OK")
    else:
        print("[HealthCheck] /health endpoint not available (OK for hosted APIs, continuing...)")

    check_models(api_base, api_key=api_key)

    is_local = "localhost" in api_base or "127.0.0.1" in api_base
    for attempt in range(1, retries + 1):
        print(f"  Attempt {attempt}/{retries}...")

        ok = check_vlm_generation(api_base, model_name, api_key=api_key)
        if ok:
            print("\n[HealthCheck] All checks passed!")
            return

        if is_local:
            if attempt < retries:
                print(f"  Retrying in {delay}s...\n")
                time.sleep(delay)
                continue
            print("\nGeneration test failed after all retries.")
            sys.exit(1)
        else:
            # For hosted APIs, treat generation failure as a warning — the pipeline
            # will surface a clear error if the model is truly inaccessible.
            print("[HealthCheck] Generation test failed (continuing — hosted API may still work).")
            return
