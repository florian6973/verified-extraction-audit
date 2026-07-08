"""Direct-identifier (DI) type registry.

The audit pipeline was originally built for a single direct identifier — the
patient *name*. As described in the paper, extending it to another direct
identifier (an MRN, an address, a phone number, ...) requires only three things:

  1. the **query prompt(s)** used to elicit the identifier from the model,
  2. the **generation length** (how many new tokens the identifier spans), and
  3. the **parsing method** that turns a raw generation into a candidate value.

This module captures exactly those three knobs per DI type, plus the two pieces
of metadata the injection step needs: which synthetic-:class:`FakePersonas`
field supplies the true value, and the semantic ``category`` tag recorded for
each blank (consumed downstream by the sampling / evaluation code).

To add a new direct identifier, add one :class:`DirectIdentifierType` entry to
:data:`DI_TYPES` (and, if needed, a new parse strategy to :data:`PARSERS`).
Nothing else in the pipeline needs to change.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import re


@dataclass(frozen=True)
class DirectIdentifierType:
    """Everything the pipeline needs to know about one kind of direct identifier."""

    name: str                  # short key, e.g. "name", "mrn", "address"
    label: str                 # note label that precedes a masked blank, e.g. "Name:"
    category: str              # semantic tag recorded for the blank (denominators/eval)
    persona_field: str         # FakePersonas column that supplies the true value
    query_prompts: List[str]   # (1) prompt(s) used for extraction / log-likelihood queries
    max_new_tokens: int        # (2) generation length when extracting this identifier
    parse_strategy: str        # (3) how to parse a raw generation (see PARSERS)
    label_aliases: Tuple[str, ...] = ()  # other note labels that mark this identifier

    @property
    def primary_prompt(self) -> str:
        return self.query_prompts[0]

    @property
    def all_labels(self) -> Tuple[str, ...]:
        return (self.label,) + tuple(self.label_aliases)


# --------------------------------------------------------------------------- #
# (3) Parsing methods: raw model generation -> candidate identifier value.
# --------------------------------------------------------------------------- #
def _parse_first_two_words(text: str) -> str:
    """First two whitespace tokens, title-cased, dots stripped (a First Last name).

    Matches the extraction heuristic used in the paper's evaluation code
    (``pipeline/paper/ner_ll_remaining.extract_names``).
    """
    tokens = text.strip().split()
    name = " ".join(tokens[:2])
    return name.replace(".", "").strip().title()


def _parse_digits(text: str) -> str:
    """The first contiguous run of digits (e.g. an MRN)."""
    match = re.search(r"\d+", text)
    return match.group(0) if match else ""


def _parse_line(text: str) -> str:
    """The first non-empty line, stripped (e.g. an address or phone)."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def _parse_word(text: str) -> str:
    """The first whitespace token (e.g. an email address)."""
    tokens = text.strip().split()
    return tokens[0] if tokens else ""


PARSERS = {
    "first_two_words": _parse_first_two_words,
    "digits": _parse_digits,
    "line": _parse_line,
    "word": _parse_word,
}


def parse_candidate(di: "DirectIdentifierType", generation: str) -> str:
    """Apply ``di``'s parse strategy to a raw model generation."""
    return PARSERS[di.parse_strategy](generation)


# --------------------------------------------------------------------------- #
# The registry. `name` mirrors the original pipeline exactly (prompts
# 'Name: '/'Patient: ', first-two-words parsing) so results are unchanged.
# --------------------------------------------------------------------------- #
DI_TYPES = {
    "name": DirectIdentifierType(
        name="name",
        label="Name:",
        category="name-patient",
        persona_field="name",
        query_prompts=["Name: ", "Patient: "],
        max_new_tokens=20,
        parse_strategy="first_two_words",
    ),
    "attending": DirectIdentifierType(
        name="attending",
        label="Attending:",
        category="name-attending",
        persona_field="physician_name",
        query_prompts=["Attending: ", "Dr. "],
        max_new_tokens=20,
        parse_strategy="first_two_words",
    ),
    "mrn": DirectIdentifierType(
        name="mrn",
        label="MRN:",
        category="id",
        persona_field="unit_no",
        query_prompts=["MRN: ", "Unit No: "],
        max_new_tokens=12,
        parse_strategy="digits",
        label_aliases=("Unit No:",),  # MIMIC labels the MRN "Unit No:"
    ),
    "address": DirectIdentifierType(
        name="address",
        label="Address:",
        category="address",
        persona_field="address",
        query_prompts=["Address: "],
        max_new_tokens=40,
        parse_strategy="line",
        label_aliases=("Mailing Address:", "Address on file:"),
    ),
    "phone": DirectIdentifierType(
        name="phone",
        label="Phone:",
        category="phone",
        persona_field="phone",
        query_prompts=["Phone: "],
        max_new_tokens=12,
        parse_strategy="line",
    ),
    "email": DirectIdentifierType(
        name="email",
        label="Email:",
        category="email",
        persona_field="email",
        query_prompts=["Email: "],
        max_new_tokens=16,
        parse_strategy="word",
    ),
}


def get_di_type(name: str) -> DirectIdentifierType:
    if name not in DI_TYPES:
        raise KeyError(
            f"Unknown direct-identifier type {name!r}. "
            f"Known types: {sorted(DI_TYPES)}. Add one to src/dataset/prepare/di_types.py."
        )
    return DI_TYPES[name]


def detect_di_type(preceding_text: str, default: DirectIdentifierType) -> DirectIdentifierType:
    """Guess the DI type of a blank from the note label just before it.

    Looks at the tail of ``preceding_text`` (the note text up to a ``___`` blank)
    for a known DI label (e.g. ``"Name:"``). Returns ``default`` if none matches,
    so notes without explicit labels fall back to the DI type chosen on the CLI.
    """
    tail = preceding_text[-40:].lower()
    best = None
    best_pos = -1
    for di in DI_TYPES.values():
        # Consider the primary label and any aliases; the rightmost-matching
        # label (closest to the blank) wins.
        for label in di.all_labels:
            pos = tail.rfind(label.lower())
            if pos > best_pos:
                best_pos = pos
                best = di
    return best if best is not None else default
