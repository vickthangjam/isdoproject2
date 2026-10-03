"""
CRRA Lab C2 - Mock Contract Management API (vibe-coded version).

Serves data/contracts.csv on http://localhost:5001 with derived policy fields.
Run from the project root:  python mcp_server/contract_shim.py
"""

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "contracts.csv"
SIMULATED_TODAY = date(2025, 4, 1)  # fixed so every run gives identical results
PATCHABLE = ("status", "owner", "proposed_uplift_pct")
INT_FIELDS = ("annual_value_inr", "notice_days", "seats_purchased",
              "seats_active", "proposed_uplift_pct")

app = Flask(__name__)
CONTRACTS: list[dict] = []


def notice_state(deadline: date, renewal: date) -> str:
    if renewal < SIMULATED_TODAY:
        return "EXPIRED"
    if deadline <= SIMULATED_TODAY:
        return "INSIDE_WINDOW"
    if (deadline - SIMULATED_TODAY).days <= 30:
        return "APPROACHING"
    return "OPEN"


def approval_band(value: int) -> str:
    if value < 1_000_000:
        return "A"
    return "B" if value <= 5_000_000 else "C"


def enrich(row: dict) -> dict:
    for field in INT_FIELDS:
        row[field] = int(row[field])
    row["auto_renew"] = row["auto_renew"].strip().upper() == "Y"
    renewal = datetime.strptime(row["renewal_date"], "%Y-%m-%d").date()
    deadline = renewal - timedelta(days=row["notice_days"])
    row["notice_deadline"] = deadline.isoformat()
    row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
    row["days_to_notice_deadline"] = (deadline - SIMULATED_TODAY).days
    row["notice_state"] = notice_state(deadline, renewal)
    seats = row["seats_purchased"]
    row["utilisation_pct"] = round(100 * row["seats_active"] / seats) if seats else None
    row["approval_band"] = approval_band(row["annual_value_inr"])
    return row


def load_contracts() -> None:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        CONTRACTS[:] = [enrich(row) for row in csv.DictReader(f)]


def find(contract_id: str) -> dict | None:
    return next((c for c in CONTRACTS
                 if c["contract_id"].upper() == contract_id.upper()), None)


def not_found(contract_id: str):
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "mock-contract-api",
                    "contracts_loaded": len(CONTRACTS),
                    "simulated_today": SIMULATED_TODAY.isoformat()})


@app.get("/api/contracts")
def list_contracts():
    results = CONTRACTS
    for param, field in (("category", "category"), ("band", "approval_band"),
                         ("notice_state", "notice_state")):
        wanted = request.args.get(param)
        if wanted:
            results = [c for c in results if c[field].lower() == wanted.lower()]
    return jsonify({"count": len(results), "contracts": results})


@app.get("/api/contracts/expiring")
def expiring():
    try:
        window = int(request.args.get("days", 90))
    except ValueError:
        return jsonify({"error": "days must be a whole number"}), 400
    results = sorted((c for c in CONTRACTS if 0 <= c["days_to_renewal"] <= window),
                     key=lambda c: c["days_to_renewal"])
    return jsonify({"count": len(results), "window_days": window, "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    contract = find(contract_id)
    return jsonify(contract) if contract else not_found(contract_id)


@app.get("/api/categories")
def categories():
    grouped: dict[str, list[dict]] = {}
    for c in CONTRACTS:
        grouped.setdefault(c["category"], []).append(
            {k: c[k] for k in ("contract_id", "vendor", "annual_value_inr", "utilisation_pct")})
    summary = [{"category": cat, "vendor_count": len(items),
                "total_annual_value_inr": sum(i["annual_value_inr"] for i in items),
                "vendors": items}
               for cat, items in sorted(grouped.items())]
    return jsonify({"count": len(summary), "categories": summary})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """In-memory only. Restarting the server resets every change."""
    contract = find(contract_id)
    if not contract:
        return not_found(contract_id)
    payload = request.get_json(silent=True) or {}
    updates = {k: payload[k] for k in PATCHABLE if k in payload}
    if not updates:
        return jsonify({"error": f"Send at least one of: {', '.join(PATCHABLE)}"}), 400
    if "proposed_uplift_pct" in updates:
        try:
            updates["proposed_uplift_pct"] = int(updates["proposed_uplift_pct"])
        except (TypeError, ValueError):
            return jsonify({"error": "proposed_uplift_pct must be a whole number"}), 400
    contract.update(updates)
    return jsonify({"updated": True, "contract": contract})


load_contracts()  # at import time, so the data is there however the app is started

if __name__ == "__main__":
    print("=" * 60)
    print(f"  Mock Contract API: {len(CONTRACTS)} contracts from {CSV_PATH.name}")
    print(f"  Simulated today: {SIMULATED_TODAY}")
    print("  http://localhost:5001/health")
    print("=" * 60)
    app.run(port=5001, debug=False)
