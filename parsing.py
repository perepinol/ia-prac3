"""Functions to load matrices for decision trees and K-means clustering."""
SUPPORTED_NUMS = [int, float]


def cast_str(value):
    """Cast a string to any of the possible types, or returns it as is."""
    for t in SUPPORTED_NUMS:
        try:
            return t(value)
        except ValueError:
            pass
    return value


def is_numerical(value):
    """Return true if value is of a supported numerical type."""
    for t in SUPPORTED_NUMS:
        if type(value) == t:
            return True
    return False


def read_file(filename, has_headers=False, rowsep='\t'):
    """
    Read matrix from file.

    Different entries should be in different lines, and values in an entry
    separated by tabs.

    If has_headers, the first row and column are returned separately
    """
    data, firstcol = [], []
    with open(filename) as fh:
        for line in fh:
            line = list(map(
                lambda x: cast_str(x),
                line.strip().split(rowsep)
            ))
            if has_headers:
                firstcol.append(line[0])
                data.append(line[1:])
            else:
                data.append(line)
    if has_headers:
        return data[0], firstcol[1:], data[1:]
    return data
