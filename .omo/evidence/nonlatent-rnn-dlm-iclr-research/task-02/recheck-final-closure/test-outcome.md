# Scoped regression

Command: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider <five Task-1 files> test_metrics.py test_prompt_masks.py`

Result: **72 passed in 1.35s**. The current collection is 72, not the supplied stale count of 71.
