"""Guardrails – Safety and compliance filters for the BFSI assistant.

Implements pre-query and post-response guardrails as specified in the PRD:
  - Reject out-of-domain queries
  - Block PII in responses
  - Prevent fabricated financial numbers
  - Add compliance disclaimers where appropriate
"""
import re
from typing import Tuple


# ── Keywords / Patterns ──────────────────────────────────────────────
BFSI_KEYWORDS = [
    "loan", "emi", "interest", "bank", "account", "deposit", "credit",
    "debit", "insurance", "policy", "premium", "claim", "mortgage",
    "savings", "current account", "fixed deposit", "fd", "rd",
    "recurring", "neft", "rtgs", "imps", "upi", "cheque", "check",
    "atm", "kyc", "aadhaar", "pan", "ifsc", "swift", "nominee",
    "pension", "mutual fund", "sip", "tax", "tds", "gst", "npa",
    "cibil", "credit score", "card", "payment", "transaction",
    "balance", "transfer", "statement", "passbook", "overdraft",
    "collateral", "guarantee", "lien", "ppf", "epf", "sukanya",
    "mudra", "jan dhan", "locker", "forex", "remittance", "gold loan",
    "vehicle loan", "car loan", "home loan", "personal loan",
    "education loan", "business loan", "rera", "ombudsman", "rbi",
    "sarfaesi", "pmjjby", "pmsby", "dicgc", "nbfc", "microfinance",
    "fintech", "digital banking", "mobile banking", "net banking",
]

HARMFUL_INTENT_PATTERNS = [
    r"\b(hack|steal|fraud|launder|illegal|evade|forge|counterfeit)\b",
    r"\b(bypass\s+kyc|fake\s+id|money\s+laundering)\b",
]

PII_PATTERNS = [
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[AADHAAR_REDACTED]"),       # Aadhaar
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN_REDACTED]"),               # PAN
    (r"\b\d{9,18}\b", "[ACCOUNT_NUMBER_REDACTED]"),               # Account numbers
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),  # Phone (US-style)
    (r"\b\d{10}\b", "[PHONE_REDACTED]"),                          # Indian phone
]

# Phrases suggesting fabricated specifics
FABRICATION_PATTERNS = [
    r"(?i)\byour\s+(exact|specific)\s+(balance|amount|rate)\s+is\b",
    r"(?i)\byour\s+account\s+number\s+is\b",
    r"(?i)\byour\s+otp\s+is\b",
]

OUT_OF_DOMAIN_RESPONSE = (
    "I'm sorry, but I can only help with banking, financial services, "
    "and insurance related queries. Could you please ask a BFSI-related "
    "question? For example, you can ask about loans, accounts, cards, "
    "insurance, or digital banking."
)

HARMFUL_RESPONSE = (
    "I'm unable to assist with that request as it may involve illegal "
    "or harmful activities. If you have a legitimate banking query, "
    "I'm happy to help."
)

DISCLAIMER = (
    "\n\n*Disclaimer: This information is for general guidance only. "
    "For specific rates, charges, or account-level details, please "
    "contact your bank branch or customer care helpline.*"
)


class Guardrails:
    """Pre- and post-processing safety filters."""

    # ── Pre-query checks ──────────────────────────────────────────────
    @staticmethod
    def check_query(query: str) -> Tuple[bool, str]:
        """Validate the user query before processing.

        Returns:
            (is_valid, message) – if not valid, message contains the
            rejection reason to show to the user.
        """
        q_lower = query.lower().strip()

        # 1. Empty query
        if not q_lower:
            return False, "Please enter a valid question."

        # 2. Harmful intent
        for pattern in HARMFUL_INTENT_PATTERNS:
            if re.search(pattern, q_lower):
                return False, HARMFUL_RESPONSE

        # 3. Domain check – at least one BFSI keyword should appear
        if not any(kw in q_lower for kw in BFSI_KEYWORDS):
            # Allow greetings / meta questions through
            greetings = ["hi", "hello", "hey", "good morning", "good evening",
                         "thanks", "thank you", "bye", "help", "what can you do"]
            if any(q_lower.startswith(g) or q_lower == g for g in greetings):
                return True, ""
            return False, OUT_OF_DOMAIN_RESPONSE

        return True, ""

    # ── Post-response checks ─────────────────────────────────────────
    @staticmethod
    def sanitise_response(response: str) -> str:
        """Clean the model's response before showing to the user."""
        # 1. Redact any PII that leaked through
        for pattern, replacement in PII_PATTERNS:
            response = re.sub(pattern, replacement, response)

        # 2. Flag fabricated-sounding specifics
        for pattern in FABRICATION_PATTERNS:
            if re.search(pattern, response):
                response = re.sub(
                    pattern,
                    "[REDACTED — contact your branch for specifics]",
                    response,
                )

        # 3. Add disclaimer for financial advice
        financial_advice_keywords = [
            "interest rate", "emi", "premium", "tax benefit",
            "loan amount", "eligibility", "credit score",
        ]
        if any(kw in response.lower() for kw in financial_advice_keywords):
            if "Disclaimer" not in response:
                response += DISCLAIMER

        return response.strip()
