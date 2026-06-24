"""Structural guards for the static UI (layout, not accessibility per se).

The staged-files list must stay a single bounded, scrollable panel. When a large
batch is added (e.g. a whole imaged book, hundreds of pages), it should scroll
inside a fixed-size box instead of growing an unbounded tower of per-file cards
that pushes the Transcribe / Estimate buttons and the progress bar far below the
fold. This mirrors the activity log under the progress bar.
"""
from fastapi.testclient import TestClient

from cursbreaker.server import app

client = TestClient(app)


def _css():
    r = client.get("/static/styles.css")
    assert r.status_code == 200
    return r.text


def _block(css, selector):
    """Return the rule body for `selector` (from the selector to its closing }), so
    assertions can't leak into adjacent rules."""
    i = css.index(selector)
    return css[i: css.index("}", i) + 1]


def test_staged_list_is_a_single_bounded_scroll_box():
    block = _block(_css(), ".staged {")
    # A capped height with its own scrollbar = one fixed-size box.
    assert "max-height:" in block, ".staged must cap its height"
    assert "overflow-y: auto" in block, ".staged must scroll when it overflows"
    # It reads as one panel (border + background), not N free-floating cards.
    assert "border:" in block and "border-radius:" in block, ".staged should be a panel"


def test_staged_rows_are_flat_not_individual_cards():
    css = _css()
    row = _block(css, ".staged li {")
    # The standalone-card look (each row its own panel background) is gone; rows
    # sit flat inside the shared box, separated by a divider instead.
    assert "background:" not in row, ".staged li should not be a standalone card"
    assert ".staged li + li" in css, "rows should be divided by a hairline"


def test_remove_button_keeps_its_44px_target_inside_the_box():
    # Regression guard: flattening the rows must not shrink the remove control
    # below the 2.5.5 target size.
    block = _block(_css(), ".staged .rm {")
    assert "min-width: 44px" in block and "min-height: 44px" in block


def test_results_offer_type_filtered_bulk_download():
    """The Results card exposes the type picker (hOCR / ALTO / PDF / text), a
    'download selected' action, and the everything-zip fallback -- so a large job
    can be fetched by type instead of one file at a time."""
    html = client.get("/").text
    for box in ('id="dl-hocr"', 'id="dl-alto"', 'id="dl-pdf"', 'id="dl-txt"'):
        assert box in html, f"missing download checkbox {box}"
    assert 'id="dl-selected"' in html and 'id="zip-link"' in html
