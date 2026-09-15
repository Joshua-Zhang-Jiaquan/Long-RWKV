def gray_to_binary(gray: int) -> int:
    """Convert a Gray-coded integer back to standard binary."""
    mask = gray >> 1
    while mask:
        gray ^= mask
        mask >>= 1
    return gray
