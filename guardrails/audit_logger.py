"""
CRRA Lab C4 - Audit Trail

Every agent decision and every human approval is recorded here, one JSON object
per line in logs/audit_trail.jsonl. The file is opened in APPEND mode on purpose:
re-running the orchestrator adds to the trail instead of replacing it. An audit
trail that a re-run can overwrite is not evidence of anything.

Delete logs/audit_trail.jsonl by hand if you want a clean file for a demo.
"""

import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []  # this run only, for the summary table

    def log(self, agent: str, action: str, contract_id: str, rationale: str = "",
            approval_status: str = "N/A", actor: str = "system") -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "action": action,
            "contract_id": contract_id,
            "rationale": rationale,
            "approval_status": approval_status,
            "actor": actor,
        }
        self.entries.append(entry)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        short = rationale if len(rationale) <= 70 else rationale[:70] + "..."
        print(f"  [AUDIT] {agent}: {action} — {short}")
        return entry

    def summary(self) -> None:
        print(f"\n{'═' * 78}\nAUDIT TRAIL — {len(self.entries)} entries this run\n{'═' * 78}")
        print(f"{'Time':<10}{'Contract':<10}{'Agent':<15}{'Action':<20}{'Approval':<10}Actor")
        print("-" * 78)
        for e in self.entries:
            print(f"{e['timestamp'][11:]:<10}{e['contract_id']:<10}{e['agent']:<15}"
                  f"{e['action'][:19]:<20}{e['approval_status']:<10}{e['actor']}")
        print("-" * 78)
        print(f"Appended to: {self.log_path}")
