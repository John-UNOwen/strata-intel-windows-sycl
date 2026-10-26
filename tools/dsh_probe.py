"""tools/dsh_probe.py - is a running strata server fit for an agent harness like DSH?

Probes three things against a server started by serve/server.py (any config), in increasing cost:
  1. shapes:      GET /health and /v1/models answer.
  2. tool calling: an OpenAI request with two tools gets a parseable tool_calls reply whose argument JSON is valid;
                  the same via /v1/messages expects a tool_use content block.
  3. tools at long context: the tool test repeated behind a ~50K-token prompt whose deep-middle line names the
                  exact city the tool argument must carry. Correct deep retrieval + a well-formed call is the shape
                  DSH sessions actually run.

Run while a server is up:  python tools/dsh_probe.py [--base http://127.0.0.1:8081] [--words 45000]
Exits 0 when every probe passes; prints PASS/FAIL lines. Nothing here loads a model itself.
"""
import argparse, json, sys, time, urllib.request

def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), time.time() - t0

def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

TOOLS = [
    {"type": "function", "function": {"name": "lookup_city",
        "description": "Report the city a line of the conversation named.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]
ANTHROPIC_TOOLS = [{"name": "lookup_city", "description": "Report the city a line of the conversation named.",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
NEEDLE = "The freight ledger names the destination city as Novosibirsk."

def prompt_words(n, needle_at=None):
    words = [f"line{i} alpha bravo charlie delta echo foxtrot" for i in range(n)]
    if needle_at is not None:
        words[needle_at] = NEEDLE
    return " ".join(words)

def ask_for_call(user):
    return ("You must call lookup_city with the city the ledger names. No prose, just the tool call.\n" + user)

def check_openai_call(msg):
    tc = (msg or {}).get("tool_calls") or []
    if not tc:
        return None
    try:
        return json.loads(tc[0]["function"]["arguments"])
    except Exception:
        return "unparseable"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--words", type=int, default=45000, help="approx long-prompt size in words")
    ap.add_argument("--model", default="", help="model name (default: whatever /v1/models reports)")
    a = ap.parse_args()
    fails = 0

    h = get(a.base, "/health")
    m = get(a.base, "/v1/models")
    model = a.model or m["data"][0]["id"]
    print(f"PASS shapes: /health {json.dumps(h)[:80]} / model {model}")

    # probe 2: tools, short context
    r, dt = post(a.base, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "system", "content": "You are a tool-calling agent."},
                                     {"role": "user", "content": ask_for_call(NEEDLE)}],
        "tools": TOOLS, "max_tokens": 64, "temperature": 0})
    args1 = check_openai_call(r["choices"][0]["message"])
    ok2 = isinstance(args1, dict) and "novosibirsk" in str(args1).lower()
    fails += 0 if ok2 else 1
    print(("PASS" if ok2 else "FAIL"), f"openai tools ({dt:.1f}s): {json.dumps(args1)[:120]}")

    r, dt = post(a.base, "/v1/messages", {
        "model": model, "system": "You are a tool-calling agent.", "max_tokens": 64, "temperature": 0,
        "messages": [{"role": "user", "content": ask_for_call(NEEDLE)}], "tools": ANTHROPIC_TOOLS})
    uses = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
    ok3 = bool(uses) and "novosibirsk" in json.dumps(uses[0].get("input", {})).lower()
    fails += 0 if ok3 else 1
    print(("PASS" if ok3 else "FAIL"), f"anthropic tools ({dt:.1f}s): {json.dumps(uses[:1])[:120]}")

    # probe 3: the same call behind a long prompt; needle 70% in
    pos = int(a.words * 0.7)
    big = prompt_words(a.words, pos)
    r, dt = post(a.base, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "system", "content": "You are a tool-calling agent."},
                                     {"role": "user", "content": ask_for_call(big)}],
        "tools": TOOLS, "max_tokens": 64, "temperature": 0})
    usage = r.get("usage", {})
    args3 = check_openai_call(r["choices"][0]["message"])
    ok4 = isinstance(args3, dict) and "novosibirsk" in str(args3).lower()
    fails += 0 if ok4 else 1
    print(("PASS" if ok4 else "FAIL"), f"tools behind {usage.get('prompt_tokens', '?')} prompt tokens "
                                       f"({dt:.1f}s): {json.dumps(args3)[:120]}")

    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
