"""Canonical Tier 1 deterministic rules shared by the dashboard and the
evaluation cascade.

Rubric: BLOCK rules match an attack *action* combined with a credential/object
belonging to someone else or left unspecified ("hack account", "steal another
person's password", "bypass account authentication"). Pure attack vocabulary
(hack, exploit, malware, attack) and defensive or ownership-scoped phrasing
recovered / defended ("recover / secure / prevent, my own account") does not
match any rule. Known residual risk, documented deliberately: protective
third-person framing such as "prevent employees from stealing our passwords"
and *educational mention* of bare "hack accounts" ("why teaching how to hack
accounts is dangerous") still match block rules; such prompts are treated
fail-closed (blocked) rather than silently allowed. Extending the exclusion
framework needs labeled cases first, not more regex.
"""

from __future__ import annotations

from .regex_filter import RegexAction, RegexFilter, RegexRule

_BLOCK_RULES = (
    RegexRule("block.ignore_previous", r"ignore\s+(all\s+)?previous"),
    RegexRule(
        "block.system_prompt",
        r"(reveal|show|print).{0,24}(system prompt|hidden instructions)",
    ),
    # Credential theft directed at another party (or unspecified owner): a
    # strong theft verb followed within a short window by a credential object.
    RegexRule(
        "block.credential_theft",
        r"\b(?:steal|grabs?|harvests?|phish(?:es)?|harvesting|stealing)\b"
        r".{0,28}\b(passwords?|credentials?|logins?|session\s+cookies?"
        r"|api\s+keys?)\b",
    ),
    # Direct unauthorized-access phrasing: bare "hack <account/object>" or an
    # explicit "hack(s|ing) into <object>".
    RegexRule(
        "block.account_takeover",
        r"\bhack\b.{0,12}\baccounts?\b",
    ),
    RegexRule(
        "block.account_takeover_into",
        r"\bhack(?:s|ing)?\s+into\b.{0,44}\baccounts?\b",
    ),
    # Bypassing an account/authentication boundary for someone else.
    RegexRule(
        "block.auth_bypass",
        r"\bbypass(?:s|ing)?\b.{0,24}\b(?:account|authentication)\b",
    ),
    # Covert-access phrasing around a credential object ("without them
    # knowing", "secretly ... passwords"), in either word order. Very rare
    # in benign phrasing.
    RegexRule(
        "block.covert_access",
        r"\b(?:secretly|covertly|stealthily|undetected"
        r"|without\s+(?:their\s+|anyone'?s?\s+)?(?:permission|consent|knowing)"
        r")\b.{0,48}\b(passwords?|credentials?|accounts?|logins?)\b"
        r"|\b(passwords?|credentials?|accounts?|logins?)\b.{0,48}\b"
        r"(?:secretly|covertly|stealthily|undetected"
        r"|without\s+(?:their\s+|anyone'?s?\s+)?(?:permission|consent|knowing)"
        r")\b",
    ),
)

_REDACT_RULES = (
    RegexRule("redact.secret", r"secret=\w+", action=RegexAction.REDACT),
    RegexRule(
        "redact.email",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        action=RegexAction.REDACT,
    ),
)

DEFAULT_TIER1_RULES = _BLOCK_RULES + _REDACT_RULES


def defense_regex_filter(max_input_bytes: int = 4096) -> RegexFilter:
    return RegexFilter(DEFAULT_TIER1_RULES, max_input_bytes=max_input_bytes)


__all__ = ["DEFAULT_TIER1_RULES", "defense_regex_filter"]
