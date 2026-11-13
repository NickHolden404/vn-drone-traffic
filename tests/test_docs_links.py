"""The front page and the results page link only to files that exist in the repository."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _local_links(page: Path) -> list[str]:
    links = LINK.findall(page.read_text(encoding="utf-8"))
    return [u for u in links if not u.startswith(("http://", "https://", "#", "mailto:"))]


@pytest.mark.parametrize("page", ["README.md", "docs/results.md"])
def test_local_links_resolve(page: str):
    p = ROOT / page
    missing = [u for u in _local_links(p) if not (p.parent / u.split("#")[0]).exists()]
    assert not missing, f"{page} links to missing files: {missing}"


def test_front_page_points_to_the_results_page():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/results.md" in text and "<!-- results:start -->" not in text
    assert "<!-- results:start -->" in (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
