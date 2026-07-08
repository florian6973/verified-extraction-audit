"""Inject direct identifiers into ``___`` blanks and build the SFT dataset.

One unified step that replaces the old fill -> classify -> manual-map -> sample
chain. For each note it:

  1. classifies every ``___`` blank into a direct-identifier category, either
     - ``--classifier label``: from the note label just before the blank
       (offline, deterministic — see :mod:`.di_types`), or
     - ``--classifier llm``: a SINGLE LLM call per note that classifies all
       blanks at once (``--api gemini|vllm|mock``);
  2. fills each blank with the matching field of the note's synthetic persona,
     at the target ``--proportion`` (the rest stay ``___``);
  3. writes the SFT JSON (train/val), a members table, and — optionally — a
     ``(entry, label)`` labeled set for the scenario-2 audit.

Personas and notes are matched by **note_id** (never by row position or file
listing order), so nothing breaks if parquet row order or ``os.listdir`` changes.

Example
-------
    python -m src.dataset.prepare.inject \
        --splits-root data/processed --version 8 --classifier label \
        --di-type name --proportion 0.05 \
        --output-sft data/processed/sft --emit-labeled data/labeled.parquet
"""

import argparse
import json
import os
import random
import re

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.dataset.prepare.di_types import DI_TYPES, detect_di_type, get_di_type

# category -> persona column that supplies the value (e.g. name-patient -> name).
CATEGORY_TO_FIELD = {di.category: di.persona_field for di in DI_TYPES.values()}
# Categories the classifier may return (fillable ones + de-identified extras).
CATEGORIES = sorted(set(CATEGORY_TO_FIELD)) + ["date", "other"]


def _blank_offsets(text):
    """Character offsets of each ``___`` blank, in document order."""
    return [m.start() for m in re.finditer("___", text)]


def _numbered(text):
    """Replace the k-th ``___`` with ``[k]`` (1-indexed) for the LLM prompt."""
    out, k = text, 0
    while "___" in out:
        k += 1
        out = out.replace("___", f"[{k}]", 1)
    return out, k


def classify_label(text, default_di):
    """{blank_number: category} from the note label preceding each blank."""
    return {str(i + 1): detect_di_type(text[:pos], default_di).category
            for i, pos in enumerate(_blank_offsets(text))}


def classify_first(text, default_di):
    """Put the DI in the FIRST blank only; every other blank stays ``___``.

    Deterministic and LLM-free — a fast way to test leakage. In MIMIC discharge
    notes the first ``___`` is the ``Name:`` field, so this injects exactly one
    identifier per note (one clean member), which the audit can then extract.
    """
    n = len(_blank_offsets(text))
    return {str(i + 1): (default_di.category if i == 0 else "other") for i in range(n)}


def classify_llm(text, note_id, api, llm_kwargs):
    """{blank_number: category} from a single LLM call classifying all blanks.

    ``llm_kwargs`` carries the backend params (``model``, and for ``openai`` the
    ``base_url`` / ``api_key``, or for ``vllm`` the ``port``).
    """
    from src.llm import call_llm

    numbered, n = _numbered(text)
    prompt = (
        "The following clinical note contains numbered de-identification blanks "
        f"[1]..[{n}]. Classify EACH blank into exactly one of these categories: "
        f"{', '.join(CATEGORIES)}. Return only a JSON object mapping each blank "
        'number (as a string) to its category, e.g. {"1": "name-patient", "2": "id"}.\n\n'
        f"Note:\n{numbered}"
    )
    resp = call_llm(api, prompt, {**llm_kwargs, "task": "classify_blanks", "note": note_id})
    return _parse_categories(resp, n)


def _coerce_obj(obj):
    """Reduce a json_repair result to a single mapping.

    Reasoning models often emit several JSON objects in one reply — restating the
    prompt's example before the final answer, or one object per blank — in which
    case ``json_repair.loads`` returns a *list*. Merge all dict members (later
    ones win, so the final answer overrides an earlier draft). Anything else -> {}.
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        merged = {}
        for o in obj:
            if isinstance(o, dict):
                merged.update(o)
        return merged
    return {}


def _parse_categories(resp, n):
    """Parse the LLM's JSON classification into {blank_number: category}.

    Tolerant of reasoning-model output: strips code fences, ignores surrounding
    reasoning prose, and merges multiple JSON objects. Falls back to parsing the
    whole text if a fenced slice yields nothing, and to ``other`` for anything
    unrecognized. Never raises.
    """
    from json_repair import json_repair
    text = resp or ""

    def _try(s):
        try:
            return _coerce_obj(json_repair.loads(s.strip()))
        except Exception:
            return {}

    obj = {}
    if "```" in text:
        # Prefer the LAST ```json block (a corrected answer wins over a first pass).
        if "```json" in text:
            fenced = text.split("```json")[-1].split("```")[0]
        elif text.count("```") >= 2:
            fenced = text.split("```")[1]
        else:
            fenced = ""
        obj = _try(fenced)
    if not obj:                       # no fence, or the fenced slice had no JSON
        obj = _try(text)

    cats = {}
    for k in range(1, n + 1):
        c = str(obj.get(str(k), "other")).strip().lower() if isinstance(obj, dict) else "other"
        cats[str(k)] = c if c in CATEGORIES else "other"
    return cats


def inject_split(filtered_df, personas_df, di, classifier, api, llm_kwargs, di_rate, rng, desc="inject"):
    """Fill blanks + sample for one split. Returns (sft_records, members).

    ``di_rate`` is the probability each direct-identifier blank keeps an identifier
    (the direct-identifier / PII rate); the rest stay ``___``.
    """
    if "note_id" not in filtered_df.columns or "note_id" not in personas_df.columns:
        raise ValueError("Both filtered and personas parquets must carry a 'note_id' column.")
    personas_by_id = personas_df.set_index("note_id")

    sft_records, members = [], []
    # tqdm shows notes/s + ETA; with --classifier llm each note is one API call,
    # so this is the live speed of name injection (progress prints to the .out).
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc=desc):
        note_id, text = row["note_id"], row["text"]
        subject_id = row.get("subject_id", "")
        if note_id not in personas_by_id.index:
            logger.warning(f"note_id {note_id} has no persona; skipping.")
            continue
        persona = personas_by_id.loc[note_id]
        if isinstance(persona, pd.DataFrame):        # duplicate note_id -> take first
            persona = persona.iloc[0]

        if classifier == "label":
            cats = classify_label(text, di)
        elif classifier == "first":
            cats = classify_first(text, di)
        else:
            cats = classify_llm(text, note_id, api, llm_kwargs)

        # Rebuild the note, filling each blank at the sampling rate.
        segments = text.split("___")
        out = segments[0]
        for k in range(1, len(segments)):
            cat = cats.get(str(k), "other")
            field = CATEGORY_TO_FIELD.get(cat)
            if field is not None and field in persona.index and rng.random() < di_rate:
                value = str(persona[field])
                out += value
                members.append({"note_id": note_id, "subject_id": subject_id,
                                "blank": k, "category": cat, "value": value})
            else:
                out += "___"      # unsampled or non-DI blank stays de-identified
            out += segments[k]

        sft_records.append({
            "instruction": "Generate a clinical note",
            "output": out,
            "note_id": note_id,
            "subject_id": subject_id,
        })
    return sft_records, members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-root", default=os.environ.get("DATA_ROOT", "data/processed"),
                        help="Root with splits_filtered_v*/ and splits_personas_v* (env DATA_ROOT)")
    parser.add_argument("--version", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--classifier", choices=["label", "first", "llm"], default="label",
                        help="How to classify each blank: 'label' note-label heuristic (offline), "
                             "'first' DI in the first blank only (offline, fast leakage test), "
                             "or 'llm' one LLM call/note")
    parser.add_argument("--api", choices=["gemini", "vllm", "openai", "mock"], default="mock",
                        help="LLM backend when --classifier llm ('openai' = any OpenAI-compatible server)")
    parser.add_argument("--model", default="mock", help="LLM model name when --classifier llm")
    parser.add_argument("--api-base", default=None,
                        help="Base URL for --api openai, e.g. http://<host>:<port>/v1 (env OPENAI_API_BASE)")
    parser.add_argument("--api-key", default=None, help="API key for --api openai (env OPENAI_API_KEY)")
    parser.add_argument("--api-port", type=int, default=None, help="Port for --api vllm (default 12346)")
    parser.add_argument("--llm-max-tokens", type=int, default=None,
                        help="Max completion tokens for --classifier llm (default backend value; "
                             "raise for reasoning models that think a lot)")
    parser.add_argument("--llm-no-think", action="store_true",
                        help="Disable model 'thinking' for --classifier llm (reasoning models: "
                             "sends chat_template_kwargs enable_thinking=false + a /no_think hint)")
    parser.add_argument("--di-type", default="name", help="Default DI type for unlabeled blanks")
    parser.add_argument("--di-rate", "--proportion", dest="di_rate", type=float, default=0.05,
                        help="Direct-identifier rate: probability each DI blank keeps an identifier "
                             "(the rest stay ___). Sets the '<split>_<rate>.json' SFT file. e.g. 0.01, 0.05, 0.1, 1.0")
    parser.add_argument("--output-sft", default=None, help="Output dir for the SFT JSON (default: <splits-root>/sft)")
    parser.add_argument("--members-dir", default=None, help="Where to write members_<split>.csv (default: <splits-root>)")
    parser.add_argument("--emit-labeled", default=None,
                        help="Write a scenario-2 (entry,label) parquet (train members=1, val members=0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    di = get_di_type(args.di_type)
    # Backend params (only what the chosen backend accepts).
    llm_kwargs = {"model": args.model}
    if args.api == "openai":
        if args.api_base:
            llm_kwargs["base_url"] = args.api_base
        if args.api_key:
            llm_kwargs["api_key"] = args.api_key
        if args.llm_max_tokens:
            llm_kwargs["max_tokens"] = args.llm_max_tokens
        if args.llm_no_think:
            llm_kwargs["no_think"] = True
    elif args.api == "vllm" and args.api_port:
        llm_kwargs["port"] = args.api_port

    output_sft = args.output_sft or os.path.join(args.splits_root, "sft")
    members_dir = args.members_dir or args.splits_root
    os.makedirs(output_sft, exist_ok=True)
    os.makedirs(members_dir, exist_ok=True)
    rng = random.Random(args.seed)

    per_split = {}
    for split in args.splits:
        filtered = os.path.join(args.splits_root, f"splits_filtered_v{args.version}", f"{split}.parquet")
        personas = os.path.join(args.splits_root, f"splits_personas_v{args.version}", f"{split}.parquet")
        if not (os.path.exists(filtered) and os.path.exists(personas)):
            logger.warning(f"Skipping {split}: missing {filtered} or {personas}")
            continue
        sft, members = inject_split(pd.read_parquet(filtered), pd.read_parquet(personas),
                                    di, args.classifier, args.api, llm_kwargs, args.di_rate, rng,
                                    desc=f"inject[{split}]")
        with open(os.path.join(output_sft, f"{split}_{args.di_rate}.json"), "w", encoding="utf-8") as f:
            json.dump(sft, f, indent=2, ensure_ascii=False)
        mdf = pd.DataFrame(members)
        mdf.to_csv(os.path.join(members_dir, f"members_{split}.csv"), index=False)
        per_split[split] = mdf
        counts = mdf["category"].value_counts().to_dict() if len(mdf) else {}
        logger.info(f"[{split}] {len(sft)} notes, injected {len(mdf)} values {counts} "
                    f"-> {output_sft}/{split}_{args.di_rate}.json")

    if args.emit_labeled and "train" in per_split:
        train = per_split["train"]
        members = sorted(set(train[train["category"] == di.category]["value"])) if len(train) else []
        non = []
        if "val" in per_split and len(per_split["val"]):
            val = per_split["val"]
            non = sorted(set(val[val["category"] == di.category]["value"]) - set(members))
        labeled = pd.DataFrame([{"entry": v, "label": 1} for v in members]
                               + [{"entry": v, "label": 0} for v in non])
        os.makedirs(os.path.dirname(args.emit_labeled) or ".", exist_ok=True)
        labeled.to_parquet(args.emit_labeled, index=False)
        logger.info(f"Wrote labeled set ({int(labeled['label'].sum())} members / "
                    f"{int((labeled['label'] == 0).sum())} non-members) -> {args.emit_labeled}")


if __name__ == "__main__":
    main()
