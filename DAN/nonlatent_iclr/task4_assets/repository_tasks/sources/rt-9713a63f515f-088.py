def tf_idf(tf: int, idf: int) -> float:
    """
    Combine the term frequency
    and inverse document frequency functions to
    calculate the originality of a term. This
    'originality' is calculated by multiplying
    the term frequency and the inverse document
    frequency : tf-idf = TF * IDF
    @params : tf, the term frequency, and idf, the inverse document
    frequency
    @examples :
    >>> tf_idf(2, 0.477)
    0.954
    """
    return round(tf * idf, 3)
