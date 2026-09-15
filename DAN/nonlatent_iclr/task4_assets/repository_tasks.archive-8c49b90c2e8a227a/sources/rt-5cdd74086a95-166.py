def determine_winner(user: str, computer: str) -> str:
    """
    Decide winner of one round.

    Parameters
    ----------
    user : str
        User's choice ("s", "w", "g").
    computer : str
        Computer's choice ("s", "w", "g").

    Returns
    -------
    str
        "user", "computer", or "draw".
    """
    if user == computer:
        return "draw"

    if user == "s" and computer == "w":
        return "computer"
    if user == "w" and computer == "s":
        return "user"

    if user == "g" and computer == "s":
        return "user"
    if user == "s" and computer == "g":
        return "computer"

    if user == "w" and computer == "g":
        return "user"
    if user == "g" and computer == "w":
        return "computer"

    return "invalid"
