"""
Static, locally-embedded MITRE ATT&CK reference (subset).

This is NOT a live feed of the MITRE ATT&CK STIX bundle (that requires
network access this sandbox doesn't have -- see docs/architecture.md).
It is a small, hand-curated subset of well-known, publicly documented
ATT&CK Enterprise technique IDs/names, used only to build a rule-based
attack graph linking the synthetic dataset's attack categories to
plausible techniques and tactics for demonstration purposes.

For a production system, replace ATTACK_TECHNIQUES with a parsed copy of
the official MITRE ATT&CK STIX bundle (https://github.com/mitre/cti),
cached under datasets/threat_intel/.
"""

from __future__ import annotations

# tactic order roughly follows the ATT&CK kill-chain progression
TACTIC_ORDER = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]

# (technique_id, technique_name, tactic)
ATTACK_TECHNIQUES = [
    ("T1595", "Active Scanning", "Reconnaissance"),
    ("T1592", "Gather Victim Host Information", "Reconnaissance"),
    ("T1590", "Gather Victim Network Information", "Reconnaissance"),
    ("T1583", "Acquire Infrastructure", "Resource Development"),
    ("T1190", "Exploit Public-Facing Application", "Initial Access"),
    ("T1133", "External Remote Services", "Initial Access"),
    ("T1110", "Brute Force", "Credential Access"),
    ("T1110.001", "Password Guessing", "Credential Access"),
    ("T1046", "Network Service Discovery", "Discovery"),
    ("T1018", "Remote System Discovery", "Discovery"),
    ("T1071", "Application Layer Protocol", "Command and Control"),
    ("T1071.001", "Web Protocols", "Command and Control"),
    ("T1105", "Ingress Tool Transfer", "Command and Control"),
    ("T1498", "Network Denial of Service", "Impact"),
    ("T1498.001", "Direct Network Flood", "Impact"),
    ("T1499", "Endpoint Denial of Service", "Impact"),
    ("T1489", "Service Stop", "Impact"),
    ("T1021", "Remote Services", "Lateral Movement"),
    ("T1021.001", "Remote Desktop Protocol", "Lateral Movement"),
    ("T1543", "Create or Modify System Process", "Persistence"),
    ("T1053", "Scheduled Task/Job", "Persistence"),
    ("T1055", "Process Injection", "Defense Evasion"),
    ("T1027", "Obfuscated Files or Information", "Defense Evasion"),
    ("T1078", "Valid Accounts", "Privilege Escalation"),
    ("T1113", "Screen Capture", "Collection"),
    ("T1041", "Exfiltration Over C2 Channel", "Exfiltration"),
]

# Rule-based mapping from the synthetic dataset's attack_cat labels to
# plausible ATT&CK technique IDs. This is an illustrative, human-authored
# mapping (not derived from any live detection signature database).
ATTACK_CAT_TO_TECHNIQUES = {
    "DoS": ["T1498", "T1498.001"],
    "DDoS": ["T1498", "T1498.001", "T1583"],
    "PortScan": ["T1595", "T1046"],
    "BruteForce": ["T1110", "T1110.001", "T1078"],
    "Exploits": ["T1190", "T1133", "T1055"],
    "Reconnaissance": ["T1595", "T1592", "T1590", "T1018"],
    "Botnet": ["T1071", "T1071.001", "T1105", "T1021"],
    "Backdoor": ["T1543", "T1053", "T1027", "T1078"],
}

TECHNIQUE_LOOKUP = {t[0]: {"name": t[1], "tactic": t[2]} for t in ATTACK_TECHNIQUES}
