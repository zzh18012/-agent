import math
import re
from dataclasses import dataclass

from mmos.competition.evidence_ids import clean_evidence_id
from mmos.schemas.competition import NumberAuditFinding, NumberAuditReport
from mmos.schemas.evidence import EvidenceRecord

PLACEHOLDER_RE = re.compile(r"\[\[ev:([^\]]+)\]\]")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_/:-])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z0-9_/:-])"
)


class NumberAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExpandedText:
    text: str
    location: str
    cited_ids: tuple[str, ...]
    allow_structural_constants: bool = False


def expand_text(
    text: str,
    records: dict[str, EvidenceRecord],
    *,
    location: str,
    allow_structural_constants: bool = False,
    seeded_ids: tuple[str, ...] = (),
) -> ExpandedText:
    cited: list[str] = []

    def replace(match: re.Match[str]) -> str:
        evidence_id = clean_evidence_id(match.group(1))
        record = records.get(evidence_id)
        if record is None:
            raise NumberAuditError(
                f"{location}: unknown evidence id in placeholder: {evidence_id}"
            )
        cited.append(evidence_id)
        return record.display_value

    expanded = PLACEHOLDER_RE.sub(replace, text)
    if PLACEHOLDER_RE.search(expanded):
        raise NumberAuditError(f"{location}: nested evidence placeholder is not allowed")
    return ExpandedText(
        text=expanded,
        location=location,
        cited_ids=tuple(dict.fromkeys([*cited, *seeded_ids])),
        allow_structural_constants=allow_structural_constants,
    )


def _allowed_values(
    fragment: ExpandedText, records: dict[str, EvidenceRecord]
) -> set[str]:
    values: set[str] = set()
    for evidence_id in fragment.cited_ids:
        record = records[evidence_id]
        values.add(record.display_value)
        values.add(str(record.raw_value))
        values.update(_NUMBER_RE.findall(record.display_value))
        values.update(_NUMBER_RE.findall(str(record.raw_value)))
        if isinstance(record.raw_value, float):
            values.add(f"{record.raw_value:.10g}")
        if isinstance(record.raw_value, int):
            values.add(str(record.raw_value))
    return values


def _token_allowed(
    token: str, allowed: set[str], *, allow_structural_constants: bool
) -> bool:
    if token in allowed:
        return True
    if allow_structural_constants and token in {"0", "1", "2", "3"}:
        return True
    try:
        value = float(token)
    except ValueError:
        return False
    for candidate in allowed:
        try:
            if math.isclose(value, float(candidate), rel_tol=0.0, abs_tol=0.0):
                return True
        except ValueError:
            continue
    return False


def audit_texts(
    fragments: list[ExpandedText],
    records: dict[str, EvidenceRecord],
    allowed_metadata_numbers: tuple[str, ...],
) -> NumberAuditReport:
    allowed_metadata = set(allowed_metadata_numbers)
    findings: list[NumberAuditFinding] = []
    checked_tokens = 0
    for fragment in fragments:
        allowed = _allowed_values(fragment, records) | allowed_metadata
        for token in _NUMBER_RE.findall(fragment.text):
            checked_tokens += 1
            if _token_allowed(
                token,
                allowed,
                allow_structural_constants=fragment.allow_structural_constants,
            ):
                continue
            findings.append(
                NumberAuditFinding(
                    token=token,
                    location=fragment.location,
                    reason=(
                        "numeric token is not an expanded evidence value and is not "
                        "an allowed problem/release metadata literal"
                    ),
                )
            )
    return NumberAuditReport(
        checked_fragments=len(fragments),
        checked_tokens=checked_tokens,
        unverified=findings,
        allowed_metadata_numbers=list(allowed_metadata_numbers),
    )


__all__ = [
    "ExpandedText",
    "NumberAuditError",
    "audit_texts",
    "expand_text",
]
