"""

Discretized :math:`\\nabla` and :math:`\\nabla^2`

"""

_SCHEMES = {
    "three-point": 2,
    "five-point": 4,
    "seven-point": 6,
    "nine-point": 8,
}


def get_scheme_order(scheme: str) -> int | None:
    return _SCHEMES.get(scheme)
