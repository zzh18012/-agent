def hypervolume_2d(
    points: list[tuple[float, float]], reference: tuple[float, float]
) -> float:
    """Area dominated by a set of 2D maximization points above the reference."""
    ref_x, ref_y = reference
    unique = sorted({(float(x), float(y)) for x, y in points})
    area = 0.0
    previous_x = ref_x
    for x, y in unique:
        if y <= ref_y:
            continue
        width = x - previous_x
        if width > 0:
            area += width * (y - ref_y)
        previous_x = x
    return area


__all__ = ["hypervolume_2d"]
