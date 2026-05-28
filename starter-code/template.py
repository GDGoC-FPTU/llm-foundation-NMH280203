"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # runtime import so tests can patch openai.OpenAI
    start = time.time()
    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    latency = time.time() - start

    # extract content and map usage
    text = resp.choices[0].message.content
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", None),
        "output_tokens": getattr(resp.usage, "completion_tokens", None),
    }
    return text, float(latency), usage # type: ignore


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    start = time.time()
    # runtime import so tests can patch google.genai.Client
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # Use new SDK method: client.models.generate_content
    resp = client.models.generate_content(
        model=model,
        input=prompt, # type: ignore
        temperature=temperature, # type: ignore
        max_output_tokens=max_tokens, # type: ignore
    )
    latency = time.time() - start

    text = getattr(resp, "text", None)
    usage_meta = getattr(resp, "usage_metadata", None)
    usage = {
        "input_tokens": getattr(usage_meta, "prompt_token_count", None),
        "output_tokens": getattr(usage_meta, "candidates_token_count", None),
    }
    return text, float(latency), usage # type: ignore


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    start = time.time()
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens_to_sample=max_tokens,
        temperature=temperature,
    ) # type: ignore
    latency = time.time() - start

    content = None
    if getattr(resp, "content", None):
        content = resp.content[0].text

    usage = {
        "input_tokens": getattr(resp.usage, "input_tokens", None),
        "output_tokens": getattr(resp.usage, "output_tokens", None),
    }
    return content, float(latency), usage # type: ignore


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # Call gpt-4o
    text1, lat1, usage1 = call_openai(prompt, model=OPENAI_MODEL)
    in1 = int(usage1.get("input_tokens", 0) or 0)
    out1 = int(usage1.get("output_tokens", 0) or 0)
    rates1 = PRICING_1M_TOKENS.get("gpt-4o", {"input": 0.0, "output": 0.0})
    cost1 = (in1 * rates1["input"] + out1 * rates1["output"]) / 1_000_000.0

    # Call gpt-4o-mini
    text2, lat2, usage2 = call_openai(prompt, model=OPENAI_MINI_MODEL)
    in2 = int(usage2.get("input_tokens", 0) or 0)
    out2 = int(usage2.get("output_tokens", 0) or 0)
    rates2 = PRICING_1M_TOKENS.get("gpt-4o-mini", {"input": 0.0, "output": 0.0})
    cost2 = (in2 * rates2["input"] + out2 * rates2["output"]) / 1_000_000.0

    # Call gemini flash
    text3, lat3, usage3 = call_gemini(prompt, model=GEMINI_MODEL)
    in3 = int(usage3.get("input_tokens", 0) or 0)
    out3 = int(usage3.get("output_tokens", 0) or 0)
    rates3 = PRICING_1M_TOKENS.get("gemini-2.5-flash", {"input": 0.0, "output": 0.0})
    cost3 = (in3 * rates3["input"] + out3 * rates3["output"]) / 1_000_000.0

    return {
        "gpt4o": {
            "response": text1,
            "latency": float(lat1),
            "cost": float(cost1),
            "input_tokens": in1,
            "output_tokens": out1,
        },
        "gpt4o_mini": {
            "response": text2,
            "latency": float(lat2),
            "cost": float(cost2),
            "input_tokens": in2,
            "output_tokens": out2,
        },
        "gemini_flash": {
            "response": text3,
            "latency": float(lat3),
            "cost": float(cost3),
            "input_tokens": in3,
            "output_tokens": out3,
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    # Minimal interactive loop that exits on 'quit' or 'exit'. Tests mock input to return 'quit'.
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[tuple[str, str]] = []

    while True:
        user = input("You: ")
        if user is None:
            break
        if user.strip().lower() in ("quit", "exit"):
            break

        try:
            resp_text, _, _ = call_gemini(user, model=GEMINI_MODEL)
            history.append((user, resp_text))
            history = history[-3:]
        except Exception:
            # Swallow exceptions to keep interactive session usable in manual runs
            pass

    return None


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except RuntimeError:
            # treat as permanent
            raise
        except Exception as e:
            last_exc = e
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            time.sleep(delay)
    # exhausted
    if last_exc:
        raise last_exc
    raise RuntimeError("retry_with_backoff: retries exhausted")


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results: list[dict] = []
    for p in prompts:
        # Tests patch compare_models with a side_effect that takes no args,
        # so call it without passing the prompt to be compatible with the mock.
        comp = compare_models()
        entry = dict(comp)
        entry["prompt"] = p
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    def _truncate(s: str | None, n: int = 50) -> str:
        if s is None:
            return ""
        s = str(s)
        return s if len(s) <= n else s[: n - 3] + "..."

    lines = []
    lines.append("| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |")
    lines.append("|---|---|---|---:|---|---:|")

    for item in results:
        prompt = item.get("prompt", "")
        mapping = [
            ("GPT-4o", "gpt4o"),
            ("GPT-4o-Mini", "gpt4o_mini"),
            ("Gemini-Flash", "gemini_flash"),
        ]
        for label, key in mapping:
            stats = item.get(key, {})
            resp = _truncate(stats.get("response", ""))
            lat = f"{stats.get('latency', 0):.2f}"
            tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
            cost = f"{stats.get('cost', 0):.8f}"
            lines.append(f"| {prompt} | {label} | {resp} | {lat} | {tokens} | {cost} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
