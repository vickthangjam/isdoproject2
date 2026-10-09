"""
CRRA Lab C4 - Portfolio Orchestrator (LangGraph)

    analysis  ->  policy_check  ->  [hitl]  ->  report

An agent may recommend, but under the procurement policy it may not commit Zensar
to anything on its own. policy_check decides, in plain Python, when a human must
sign off; hitl asks that human; every step is written to the audit trail.

Prerequisites:
    1. .env in the project root containing ANTHROPIC_API_KEY=...
    2. python mcp_server/contract_shim.py   (Lab C2, running in another terminal)

Run from the project root:
    python orchestrator/supervisor.py                    # CTR-1010, CTR-1012, CTR-1006
    python orchestrator/supervisor.py CTR-1005           # any contracts you choose
"""

import json
import os
import sys
from pathlib import Path
from typing import TypedDict

# `python orchestrator/supervisor.py` only puts orchestrator/ on the import path.
# Add the project root first, so guardrails/ and data/ can be imported.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic  # noqa: E402
import chromadb  # noqa: E402
import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from guardrails.audit_logger import AuditLogger  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

MODEL = "claude-opus-5"
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"
KB_DIR = PROJECT_ROOT / "data" / "kb"
MAX_ROUNDS = 5

DEFAULT_PORTFOLIO = ["CTR-1010", "CTR-1012", "CTR-1006"]
NO_DECISION = "NO_DECISION"  # used when the agent could not produce a recommendation

audit = AuditLogger()
client = None  # created in main(), once the key has been checked
KB = None      # built in main()


def extract_text(response) -> str:
    """First block with text. A thinking block may come first."""
    for block in response.content:
        if hasattr(block, "text") and block.text:
            return block.text.strip()
    return ""


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════

class ContractState(TypedDict):
    contract_id: str
    contract: dict

    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int

    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str

    final_status: str


def initial_state(contract_id: str) -> ContractState:
    return {
        "contract_id": contract_id, "contract": {}, "recommendation": "", "confidence": "",
        "rationale": "", "policy_citation": "", "estimated_annual_impact_inr": 0,
        "hitl_required": False, "hitl_reason": "", "hitl_approved": False,
        "approver": "", "final_status": "",
    }


# ══════════════════════════════════════════════════════════════
# POLICY KB (Lab C1)
# ══════════════════════════════════════════════════════════════

def load_kb():
    """C1's ChromaDB is in-memory, so rebuild the collection here from data/kb/.
    Cosine distance, so that "1 - distance" is a sensible 0-1 confidence
    (Chroma's default L2 distance makes it negative)."""
    from data.kb_setup import chunk_article

    kb_client = chromadb.Client()
    collection = kb_client.create_collection(KB_COLLECTION, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for md in sorted(KB_DIR.glob("*.md")):
        for chunk in chunk_article(md.read_text(encoding="utf-8"), md.name):
            ids.append(chunk["id"])
            docs.append(chunk["document"])
            metas.append(chunk["metadata"])
    if not ids:
        raise SystemExit(f"No policy articles found in {KB_DIR}")
    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Policy KB ready: {len(ids)} sections")
    return collection


def search_policy(query: str) -> list[dict]:
    """Best section per policy file, top two files."""
    res = KB.query(query_texts=[query], n_results=6)
    best: dict[str, tuple[float, str, str]] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        src = meta["source"]
        if src not in best or dist < best[src][0]:
            best[src] = (dist, meta["heading"], doc)
    top = sorted(best.items(), key=lambda kv: kv[1][0])[:2]
    return [{"source": s, "section": h, "confidence": round(1 - d, 2), "text": t}
            for s, (d, h, t) in top]


# ══════════════════════════════════════════════════════════════
# NODE 1 - ANALYSIS (the Lab C3 agent as one node)
# ══════════════════════════════════════════════════════════════

ANALYSIS_TOOLS = [
    {
        "name": "search_policy",
        "description": ("Search the Zensar BizOps procurement policy. Returns the best section "
                        "from each of the top two policy files. Call at most twice."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "Plain-English policy question"}},
            "required": ["query"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation. Call exactly once, as the last step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {"type": "string",
                                   "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {"type": "string",
                              "description": "Two or three sentences naming the numbers that drove it."},
                "policy_citation": {"type": "string",
                                    "description": "Policy file and section, e.g. 'renegotiation_levers.md §Price uplift benchmarks'"},
                "estimated_annual_impact_inr": {"type": "integer",
                                                "description": "Rough annual impact in INR. Negative = saving."},
            },
            "required": ["recommendation", "confidence", "rationale",
                         "policy_citation", "estimated_annual_impact_inr"],
        },
    },
]

ANALYSIS_SYSTEM = """You are the Renewal Analysis Agent for Zensar BizOps.

You are given one contract's facts. Recommend exactly one of RENEW, RENEGOTIATE,
CONSOLIDATE or TERMINATE. Search the policy (at most twice) to find the governing rule,
then call submit_recommendation exactly once.

- RENEW: healthy utilisation and a modest uplift.
- RENEGOTIATE: still needed but the terms are wrong. An uplift above 15% is never accepted at first offer.
- CONSOLIDATE: still needed, but another vendor in the same category can cover it
  (typically utilisation below 40%).
- TERMINATE: nobody needs the capability at all. An UNASSIGNED owner is not evidence of
  that; it means nobody has been asked. Recommend conservatively and say why.

You do not decide whether a human must approve. A separate policy check does that.

LOW confidence is a valid, useful answer. Say so plainly instead of inventing certainty,
and do not keep searching hoping for a cleaner picture.

Always cite the policy file and section that supports your recommendation."""

CONTRACT_FACTS = ("contract_id", "vendor", "category", "business_unit", "owner",
                  "annual_value_inr", "approval_band", "renewal_date", "notice_deadline",
                  "days_to_notice_deadline", "notice_state", "auto_renew", "seats_purchased",
                  "seats_active", "utilisation_pct", "proposed_uplift_pct")


def fetch_contract(cid: str) -> tuple[dict | None, str]:
    """Returns (contract, error_status)."""
    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{cid}", timeout=10)
    except requests.exceptions.RequestException:
        return None, "ERROR_API_UNREACHABLE"
    if r.status_code == 404:
        return None, "ERROR_NOT_FOUND"
    if not r.ok:
        return None, f"ERROR_HTTP_{r.status_code}"
    return r.json(), ""


def no_decision(state: ContractState, contract: dict, reason: str) -> ContractState:
    print(f"  ⚠️  No recommendation: {reason}")
    audit.log("AnalysisAgent", "analysis_incomplete", state["contract_id"], reason)
    return {**state, "contract": contract, "recommendation": NO_DECISION, "confidence": "LOW",
            "rationale": reason, "policy_citation": "n/a", "estimated_annual_impact_inr": 0}


def analysis_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    print(f"\n{'═' * 66}\nCONTRACT: {cid}\n{'═' * 66}\n\n▶ ANALYSIS AGENT")

    contract, error = fetch_contract(cid)
    if error:
        print(f"  ✗ {error}")
        audit.log("AnalysisAgent", "fetch_failed", cid, error)
        return {**state, "final_status": error}

    util = contract["utilisation_pct"]
    print(f"  {contract['vendor']} · {contract['category']} · band {contract['approval_band']} · "
          f"owner {contract['owner']}\n"
          f"  INR {contract['annual_value_inr']:,}/yr · util {'n/a' if util is None else f'{util}%'} · "
          f"uplift {contract['proposed_uplift_pct']}% · {contract['notice_state']}")

    facts = json.dumps({k: contract[k] for k in CONTRACT_FACTS}, indent=2)
    messages = [{"role": "user", "content": f"Analyse this contract:\n{facts}"}]

    for _ in range(MAX_ROUNDS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                output_config={"effort": "medium"},  # temperature is not supported on this model
                system=ANALYSIS_SYSTEM,
                tools=ANALYSIS_TOOLS,
                messages=messages,
            )
        except anthropic.AuthenticationError as e:
            raise SystemExit(f"API key rejected (HTTP 401): {e.message}")
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            return no_decision(state, contract, f"Anthropic API error: {e}")

        if response.stop_reason != "tool_use":
            text = extract_text(response)
            return no_decision(state, contract,
                               text[:300] or f"Model stopped ({response.stop_reason}).")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "submit_recommendation":
                a = block.input
                print(f"  → {a['recommendation']} (confidence {a['confidence']}) · {a['policy_citation']}")
                audit.log("AnalysisAgent", "recommendation", cid,
                          f"{a['recommendation']} / {a['confidence']} — {a['policy_citation']}")
                return {**state, "contract": contract,
                        "recommendation": a["recommendation"], "confidence": a["confidence"],
                        "rationale": a["rationale"], "policy_citation": a["policy_citation"],
                        "estimated_annual_impact_inr": a.get("estimated_annual_impact_inr") or 0}
            if block.name == "search_policy":
                hits = search_policy(block.input["query"])
                print(f'  → policy "{block.input["query"][:44]}" -> ' +
                      ", ".join(f"{h['source']} §{h['section']} ({h['confidence']:.0%})" for h in hits))
                result = {"results": hits}
            else:
                result = {"error": f"Unknown tool {block.name}"}
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result), "is_error": "error" in result})
        messages.append({"role": "user", "content": tool_results})

    return no_decision(state, contract, f"No recommendation after {MAX_ROUNDS} rounds.")


# ══════════════════════════════════════════════════════════════
# NODE 2 - POLICY CHECK (pure Python, no model call)
# ══════════════════════════════════════════════════════════════

def approval_triggers(contract: dict, recommendation: str, confidence: str) -> list[str]:
    """Every rule that forces a human. Accumulates; never stops at the first match."""
    reasons = []
    band = contract.get("approval_band")
    if band in ("B", "C"):
        reasons.append(f"approval band {band} requires a named human approver")
    if contract.get("notice_state") == "INSIDE_WINDOW":
        reasons.append("inside the notice window — leverage already lost")
    if recommendation == "TERMINATE":
        reasons.append("all terminations require written owner confirmation")
    if confidence == "LOW":
        reasons.append("analysis confidence is LOW")
    if str(contract.get("owner", "")).strip().upper() == "UNASSIGNED":
        reasons.append("no business owner on record")

    # Step 6 (your exercise): auto-renewing contract approaching its notice deadline.
    # Doing nothing here has a permanent consequence: another full term at the new price.
    if contract.get("auto_renew") and contract.get("notice_state") == "APPROACHING":
        reasons.append(
            f"auto-renews on {contract['renewal_date']} unless notice is given by "
            f"{contract['notice_deadline']} ({contract['days_to_notice_deadline']} days away) — "
            f"doing nothing commits Zensar to another term at a {contract['proposed_uplift_pct']}% uplift"
        )
    return reasons


def policy_check_node(state: ContractState) -> ContractState:
    if state.get("final_status", "").startswith("ERROR"):
        return state  # nothing to check; report will record the error
    print("\n▶ POLICY CHECK")
    reasons = approval_triggers(state["contract"], state["recommendation"], state["confidence"])
    reason_text = "; ".join(reasons) if reasons else "no policy trigger"

    print(f"  HITL required: {bool(reasons)}")
    for r in reasons:
        print(f"    · {r}")
    audit.log("PolicyCheck", "evaluate_triggers", state["contract_id"], reason_text)
    return {**state, "hitl_required": bool(reasons), "hitl_reason": reason_text}


# ══════════════════════════════════════════════════════════════
# NODE 3 - HUMAN APPROVAL GATE
# ══════════════════════════════════════════════════════════════

def ask(prompt: str, valid: dict[str, bool] | None = None):
    """input() that re-asks until it gets a usable answer."""
    while True:
        answer = input(prompt).strip()
        if valid is None and answer:
            return answer
        if valid is not None and answer.lower() in valid:
            return valid[answer.lower()]
        print("    Please answer y or n." if valid else "    A name is required for the audit trail.")


def hitl_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    print("\n▶ HUMAN APPROVAL GATE")
    print(f"  Contract : {cid}  ({state['contract'].get('vendor')})")
    print(f"  Proposed : {state['recommendation']}  (confidence {state['confidence']})")
    print("  Because  :")
    for r in state["hitl_reason"].split("; "):
        print(f"    · {r}")
    print(f"  Policy   : {state['policy_citation']}")
    print(f"\n  {state['rationale']}")

    audit.log("HITLGate", "approval_request", cid,
              f"{state['recommendation']} — {state['hitl_reason']}", "PENDING")

    approved = ask("\n  Approve this recommendation? [y/n]: ",
                   {"y": True, "yes": True, "n": False, "no": False})
    # A rejection is a decision too, so it gets a named actor as well
    approver = ask("  Your name (approver): " if approved else "  Your name (rejecting): ")

    status = "APPROVED" if approved else "REJECTED"
    audit.log("HITLGate", "approval_decision", cid,
              f"{state['recommendation']} {status.lower()} by {approver}", status, actor=approver)
    print(f"  → {status} by {approver}" + ("" if approved else " — no action will be taken"))
    return {**state, "hitl_approved": approved, "approver": approver}


# ══════════════════════════════════════════════════════════════
# NODE 4 - REPORT
# ══════════════════════════════════════════════════════════════

def report_node(state: ContractState) -> ContractState:
    print("\n▶ REPORT")
    cid = state["contract_id"]
    rec = state.get("recommendation")

    if state.get("final_status", "").startswith("ERROR"):
        final, note, status = state["final_status"], "Could not analyse this contract.", "N/A"
    elif not state.get("hitl_required"):
        final, note, status = f"{rec}_AUTO", "Actioned without human approval (no policy trigger).", "N/A"
    elif state.get("hitl_approved") and rec == NO_DECISION:
        final, note, status = "ON_HOLD_MANUAL_REVIEW", \
            f"No recommendation to action; {state['approver']} took it for manual review.", "APPROVED"
    elif state.get("hitl_approved"):
        final, note, status = f"{rec}_APPROVED", \
            f"Approved by {state['approver']}. Cleared to action.", "APPROVED"
    else:
        final, note, status = "ON_HOLD_REJECTED", \
            f"Rejected by {state['approver']}. No commitment made to the vendor.", "REJECTED"

    audit.log("Reporting", "final_status", cid, f"{final} — {note}", status,
              actor=state.get("approver") or "system")
    print(f"  FINAL STATUS: {final}\n  {note}")
    return {**state, "final_status": final}


# ══════════════════════════════════════════════════════════════
# GRAPH
# ══════════════════════════════════════════════════════════════

def route_after_policy_check(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


def build_graph():
    g = StateGraph(ContractState)
    g.add_node("analysis", analysis_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("hitl", hitl_node)
    g.add_node("report", report_node)

    g.add_edge(START, "analysis")
    g.add_edge("analysis", "policy_check")
    g.add_conditional_edges("policy_check", route_after_policy_check,
                            {"hitl": "hitl", "report": "report"})
    g.add_edge("hitl", "report")
    g.add_edge("report", END)
    return g.compile()


def main() -> None:
    global client, KB
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. Add it to the .env file in the project root.")
    try:
        requests.get(f"{CONTRACT_API}/health", timeout=5)
    except requests.exceptions.RequestException:
        raise SystemExit("Contract API unreachable. Start it in another terminal first:\n"
                         "    python mcp_server/contract_shim.py")

    client = anthropic.Anthropic()
    KB = load_kb()
    graph = build_graph()

    portfolio = [a.upper() for a in sys.argv[1:]] or DEFAULT_PORTFOLIO
    outcomes = [graph.invoke(initial_state(cid)) for cid in portfolio]

    print(f"\n\n{'═' * 66}\nPORTFOLIO REVIEW COMPLETE\n{'═' * 66}")
    print(f"{'Contract':<11}{'Action':<14}{'Conf':<8}{'Gate':<8}{'Final':<26}Approver")
    print("-" * 66)
    for o in outcomes:
        gate = "HUMAN" if o.get("hitl_required") else "auto"
        print(f"{o['contract_id']:<11}{o.get('recommendation') or '—':<14}"
              f"{o.get('confidence') or '—':<8}{gate:<8}{o.get('final_status') or '—':<26}"
              f"{o.get('approver') or ''}")
    audit.summary()


if __name__ == "__main__":
    main()
