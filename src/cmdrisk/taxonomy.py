"""Pinned ATT&CK v15.1 tactic metadata and display risk weights."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tactic:
    id: str
    name: str
    severity: float


TACTICS: tuple[Tactic, ...] = (
    Tactic("TA0043", "Reconnaissance", 0.60),
    Tactic("TA0042", "Resource Development", 0.60),
    Tactic("TA0001", "Initial Access", 0.80),
    Tactic("TA0002", "Execution", 0.80),
    Tactic("TA0003", "Persistence", 0.90),
    Tactic("TA0004", "Privilege Escalation", 0.90),
    Tactic("TA0005", "Defense Evasion", 0.90),
    Tactic("TA0006", "Credential Access", 1.00),
    Tactic("TA0007", "Discovery", 0.60),
    Tactic("TA0008", "Lateral Movement", 1.00),
    Tactic("TA0009", "Collection", 0.90),
    Tactic("TA0011", "Command and Control", 1.00),
    Tactic("TA0010", "Exfiltration", 1.00),
    Tactic("TA0040", "Impact", 1.00),
)

TACTIC_BY_ID = {item.id: item for item in TACTICS}

# The model's vocabulary is data-derived. This map is the canonical fallback
# used for display, Teacher normalisation, and fixture generation.
TECHNIQUE_NAMES: dict[str, str] = {
    "T1003": "OS Credential Dumping",
    "T1005": "Data from Local System",
    "T1016": "System Network Configuration Discovery",
    "T1021": "Remote Services",
    "T1027": "Obfuscated Files or Information",
    "T1033": "System Owner/User Discovery",
    "T1036": "Masquerading",
    "T1037": "Boot or Logon Initialization Scripts",
    "T1041": "Exfiltration Over C2 Channel",
    "T1046": "Network Service Discovery",
    "T1053": "Scheduled Task/Job",
    "T1057": "Process Discovery",
    "T1059": "Command and Scripting Interpreter",
    "T1069": "Permission Groups Discovery",
    "T1070": "Indicator Removal",
    "T1071": "Application Layer Protocol",
    "T1082": "System Information Discovery",
    "T1083": "File and Directory Discovery",
    "T1087": "Account Discovery",
    "T1098": "Account Manipulation",
    "T1105": "Ingress Tool Transfer",
    "T1110": "Brute Force",
    "T1140": "Deobfuscate/Decode Files or Information",
    "T1222": "File and Directory Permissions Modification",
    "T1486": "Data Encrypted for Impact",
    "T1489": "Service Stop",
    "T1490": "Inhibit System Recovery",
    "T1548": "Abuse Elevation Control Mechanism",
    "T1562": "Impair Defenses",
    "T1564": "Hide Artifacts",
    "T1590": "Gather Victim Network Information",
    "T1053.003": "Cron",
    "T1059.004": "Unix Shell",
    "T1424": "Process Discovery",
}

TECHNIQUE_TO_TACTICS: dict[str, tuple[str, ...]] = {
    "T1003": ("TA0006",),
    "T1005": ("TA0009",),
    "T1016": ("TA0007",),
    "T1021": ("TA0008",),
    "T1027": ("TA0005",),
    "T1033": ("TA0007",),
    "T1036": ("TA0005",),
    "T1037": ("TA0003", "TA0004"),
    "T1041": ("TA0010",),
    "T1046": ("TA0007",),
    "T1053": ("TA0002", "TA0003", "TA0004"),
    "T1057": ("TA0007",),
    "T1059": ("TA0002",),
    "T1069": ("TA0007",),
    "T1070": ("TA0005",),
    "T1071": ("TA0011",),
    "T1082": ("TA0007",),
    "T1083": ("TA0007",),
    "T1087": ("TA0007",),
    "T1098": ("TA0003", "TA0004"),
    "T1105": ("TA0011",),
    "T1110": ("TA0006",),
    "T1140": ("TA0005",),
    "T1222": ("TA0005",),
    "T1486": ("TA0040",),
    "T1489": ("TA0040",),
    "T1490": ("TA0040",),
    "T1548": ("TA0004", "TA0005"),
    "T1562": ("TA0005",),
    "T1564": ("TA0005",),
    "T1590": ("TA0043",),
    "T1053.003": ("TA0002", "TA0003", "TA0004"),
    "T1059.004": ("TA0002",),
    "T1424": ("TA0007",),
}


def normalize_tactic(value: str) -> str | None:
    token = value.strip().upper()
    if token in TACTIC_BY_ID:
        return token
    for tactic in TACTICS:
        if tactic.name.casefold() == value.strip().casefold():
            return tactic.id
    return None


def normalize_technique(value: str) -> str | None:
    token = value.strip().upper()
    if token in TECHNIQUE_NAMES:
        return token
    for technique in TECHNIQUE_NAMES:
        if TECHNIQUE_NAMES[technique].casefold() == value.strip().casefold():
            return technique
    return None


def tactics_for_techniques(techniques: list[str]) -> list[str]:
    result = {
        tactic_id
        for technique in techniques
        for tactic_id in TECHNIQUE_TO_TACTICS.get(technique, ())
    }
    return [tactic.id for tactic in TACTICS if tactic.id in result]


def attack_stage(tactics: list[str]) -> str:
    supported = [TACTIC_BY_ID[item] for item in tactics if item in TACTIC_BY_ID]
    if not supported:
        return "unknown"
    return max(supported, key=lambda item: (item.severity, item.id)).name


def stage_severity(tactics: list[str]) -> float:
    supported = [TACTIC_BY_ID[item].severity for item in tactics if item in TACTIC_BY_ID]
    return max(supported, default=0.0)
