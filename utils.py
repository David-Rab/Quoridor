# helpers ----------------------------------------------------------
def to_idx(r: int, c: int, n: int) -> int:  # row-major index
    return r * n + c


def to_rc(idx: int, n: int) -> tuple[int, int]:
    return divmod(idx, n)  # cheaper than (//, %)
