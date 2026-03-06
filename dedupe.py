def size_tolerance_group(files, tolerance=0.05):
    """
    Group files by similar size (within ±5% of each other).
    Sorting first brings this from O(n²) down to O(n log n).
    Only files in the same group are compared further.
    """
    if not files:
        return []

    sorted_files = sorted(files, key=lambda f: f.size)
    groups = [[sorted_files[0]]]

    for file in sorted_files[1:]:
        ref = groups[-1][0]
        lower = ref.size * (1 - tolerance)
        upper = ref.size * (1 + tolerance)

        if lower <= file.size <= upper:
            groups[-1].append(file)
        else:
            groups.append([file])

    return groups


def find_exact_duplicate(new_file, existing_files):
    """
    Given a newly uploaded file and a list of existing DB file records,
    return the first exact duplicate (matching full hash) or None.
    """
    for existing in existing_files:
        if existing.full_hash and existing.full_hash == new_file.full_hash:
            return existing
    return None
