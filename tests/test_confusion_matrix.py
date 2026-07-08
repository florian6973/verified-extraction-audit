"""The experimental extraction confusion matrix must treat every non-member
candidate (labeled 'val' + unlabeled 'other') as a negative — matching the
paper's bootstrap_metrics (y_true = groundtruth == 'train'). Regression guard for
the bug where FPR was computed over only the labeled non-members (usually ~0).
"""

import numpy as np

from src.dataset.prepare.di_types import get_di_type
from src.evaluation.audit.from_labels import _confusion, _extract, _looks_like_name, _parse


def test_position_match_requires_name_at_index_1():
    di = get_di_type("name")
    # clean " First Last ..." right after the prompt -> name at idx 1 -> kept
    assert _extract(di, " Donald Walker MRN: 5", strict=False, position_match=True) == "Donald Walker"
    # no leading space (idx 0) or extra leading space (idx 2) -> rejected
    assert _extract(di, "Donald Walker", strict=False, position_match=True) is None
    assert _extract(di, "  Donald Walker", strict=False, position_match=True) is None
    # position_match off -> always returns the first-two-words parse
    assert _extract(di, "Donald Walker", strict=False, position_match=False) == "Donald Walker"


def test_strict_parse_preserves_case():
    di = get_di_type("name")
    # default: .title() collapses casings -> both map to the same member
    assert _parse(di, "donald walker", strict=False) == "Donald Walker"
    assert _parse(di, "DONALD WALKER", strict=False) == "Donald Walker"
    # strict: exact case preserved -> lowercase no longer matches "Donald Walker"
    assert _parse(di, "donald walker", strict=True) == "donald walker"
    assert _parse(di, "Donald Walker", strict=True) == "Donald Walker"
    # both take first two words and strip dots
    assert _parse(di, "Amy Romero from ED", strict=True) == "Amy Romero"


def test_name_filter_drops_junk_keeps_names():
    # real names survive
    assert _looks_like_name("Donald Walker")
    assert _looks_like_name("Margaret Johnson")
    # de-id placeholders / junk dropped
    assert not _looks_like_name("___ ___")
    assert not _looks_like_name("___ Unit")
    assert not _looks_like_name("Super 8")        # digit
    assert not _looks_like_name("Walker")          # single token
    assert not _looks_like_name("Dr. Smith, MD")   # comma
    assert not _looks_like_name("")
    assert not _looks_like_name(None)


def test_members_vs_all_others():
    # 4 members (2 pass tau), 6 non-members (3 pass tau). total injected members = 10.
    is_member = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    passed = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
    m = _confusion(is_member, passed, total_members=10)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 3, 2, 3)
    assert m["tpr"] == 2 / 4      # recall on the 4 generated members
    assert m["fpr"] == 3 / 6      # 3 of 6 non-members flagged — NOT ~0
    assert m["ppv"] == 2 / 5      # precision over the 5 flagged
    assert m["recall_with_verification"] == 2 / 10   # TP / all injected members


def test_no_members_still_reports_fpr():
    # The degenerate case that produced the bug: no members regenerated, but
    # non-members DO pass -> FPR must be measured, not None/0.
    is_member = np.array([0, 0, 0, 0], dtype=bool)
    passed = np.array([1, 0, 1, 0], dtype=bool)
    m = _confusion(is_member, passed, total_members=225)
    assert m["tp"] == 0 and m["fp"] == 2 and m["tn"] == 2
    assert m["fpr"] == 0.5
    assert m["tpr"] is None            # no members in the stream
    assert m["ppv"] == 0.0
    assert m["recall_with_verification"] == 0.0


def test_counts_partition_the_stream():
    rng = np.random.default_rng(0)
    is_member = rng.random(1000) < 0.1
    passed = rng.random(1000) < 0.4
    m = _confusion(is_member, passed, total_members=500)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 1000
    assert m["fpr"] == m["fp"] / (m["fp"] + m["tn"])
