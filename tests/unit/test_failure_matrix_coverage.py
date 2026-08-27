import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROW = re.compile(r"^\|\s*([PTIKRO]\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|", re.MULTILINE)


def test_every_failure_matrix_row_has_an_explicit_release_disposition() -> None:
    matrix = (ROOT / "docs/safety-model.md").read_text(encoding="utf-8")
    coverage = (ROOT / "docs/safety-coverage.md").read_text(encoding="utf-8")
    expected = {match.group(1) for match in ROW.finditer(matrix)}
    rows = {match.group(1): match.groups()[1:] for match in ROW.finditer(coverage)}
    assert set(rows) == expected
    for identifier, (disposition, reason) in rows.items():
        assert disposition.strip() in {"IMPLEMENTED", "PARTIAL", "DEFERRED"}, identifier
        assert len(reason.strip()) >= 20, identifier
