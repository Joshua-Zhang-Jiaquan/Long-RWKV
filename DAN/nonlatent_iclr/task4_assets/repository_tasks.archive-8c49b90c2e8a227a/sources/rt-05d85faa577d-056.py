def build_toy_dataset() -> tuple[list[str], list[str]]:
    """
    Build a tiny text dataset for examples and quick local testing.

    >>> texts, labels = build_toy_dataset()
    >>> len(texts), len(labels)
    (6, 6)
    >>> sorted(set(labels))
    ['ham', 'spam']
    """
    texts = [
        "buy cheap meds now",
        "cheap meds available online",
        "win cash prizes now",
        "project meeting schedule attached",
        "let us discuss the project timeline",
        "team meeting moved to monday",
    ]
    labels = ["spam", "spam", "spam", "ham", "ham", "ham"]
    return texts, labels
