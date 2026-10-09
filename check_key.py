"""
CRRA Lab C3 - API key diagnostic. Prints no secrets.

Run from the project root:
    python check_key.py
"""

import os
from pathlib import Path

import anthropic
from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"


def describe_key(k: str) -> str:
    if not k:
        return "(empty)"
    issues = []
    if k != k.strip():
        issues.append("has spaces or a newline at the ends")
    if k[0] in "\"'" or k[-1] in "\"'":
        issues.append("is wrapped in quotes")
    if not k.strip("\"' ").startswith("sk-ant-api"):
        issues.append("does not start with sk-ant-api")
    return f"starts {k[:10]!r}, ends ...{k[-4:]!r}, length {len(k)}" + (
        f"  <-- PROBLEM: {', '.join(issues)}" if issues else "")


print("=" * 60)
print("1. Settings already in Windows (these win over .env)")
for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
             "HTTPS_PROXY", "HTTP_PROXY"):
    val = os.environ.get(name)
    if not val:
        print(f"   {name:<22} not set")
    elif name == "ANTHROPIC_API_KEY":
        print(f"   {name:<22} SET: {describe_key(val)}")
    elif name == "ANTHROPIC_AUTH_TOKEN":
        print(f"   {name:<22} SET  <-- PROBLEM: the SDK sends this too; remove it")
    else:
        print(f"   {name:<22} SET: {val}")

print("\n2. The key in .env")
if not ENV_FILE.exists():
    print(f"   {ENV_FILE} does not exist")
else:
    file_key = dotenv_values(ENV_FILE).get("ANTHROPIC_API_KEY") or ""
    print(f"   {describe_key(file_key)}")
    win_key = os.environ.get("ANTHROPIC_API_KEY")
    if win_key and win_key != file_key:
        print("   PROBLEM: Windows has a DIFFERENT key, and that is the one being used.")
        print("   Fix for this terminal:  Remove-Item Env:ANTHROPIC_API_KEY")

print("\n3. Asking the Anthropic API (free call, lists models)")
load_dotenv(ENV_FILE)
print(f"   anthropic SDK version {anthropic.__version__}")
try:
    client = anthropic.Anthropic()
    ids = [m.id for m in client.models.list(limit=100)]
    print(f"   OK: key accepted, {len(ids)} models visible")
    print(f"   claude-opus-5 available: {'claude-opus-5' in ids}")
except anthropic.APIStatusError as e:
    print(f"   HTTP {e.status_code}: {e.message}")
    print(f"   request id: {getattr(e, 'request_id', None)}")
except anthropic.APIConnectionError as e:
    print(f"   Could not connect: {e!r}")
    print("   Likely a company proxy or firewall blocking api.anthropic.com.")
print("=" * 60)
