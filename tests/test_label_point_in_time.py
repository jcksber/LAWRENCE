"""
Guards against the #1 way trading-bot backtests lie: a feature that peeks
past its label's exit_timestamp.

This starts as a scaffold — fill in `compute_label` and `compute_features`
imports once src/features/ exists, then flesh out these cases. Keep this
test file growing as new features are added; every new feature column
should get a corresponding "does not leak the future" case here.
"""
import pytest


def test_scaffold_placeholder():
    """Replace with real point-in-time-correctness tests once
    src/features/label.py and src/features/gold.py exist (Phase 2)."""
    pytest.skip("Phase 2: implement once label + feature functions exist")


# Planned cases (Phase 2):
# - test_label_uses_only_entry_and_exit_timestamps()
# - test_liquidity_feature_snapshot_is_at_or_before_entry()
# - test_social_feature_excludes_posts_after_entry_timestamp()
# - test_rug_before_exit_clamps_return_at_pull_event()
