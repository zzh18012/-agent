def clean_evidence_id(value: str) -> str:
    candidate = value.strip()
    if candidate.startswith("[[ev:") and candidate.endswith("]]"):
        candidate = candidate[5:-2].strip()
    while candidate.startswith("ev:"):
        candidate = candidate[3:].strip()
    return candidate


def clean_evidence_ids(values: list[str]) -> list[str]:
    return list(dict.fromkeys(clean_evidence_id(value) for value in values))


__all__ = ["clean_evidence_id", "clean_evidence_ids"]
