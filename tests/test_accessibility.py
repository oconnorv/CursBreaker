"""WCAG 2.1 AAA guards for the static UI.

The contrast checks parse the real colour tokens out of styles.css and compute
the ratios, so a future colour tweak that drops below the AAA 7:1 threshold fails
here rather than shipping. The structural checks lock in the non-contrast AAA
work (target size, abbreviation expansions, landmarks, plain-language glossary).
"""
import re

from fastapi.testclient import TestClient

from cursbreaker.server import app

client = TestClient(app)

AAA_NORMAL = 7.0  # WCAG 1.4.6 Contrast (Enhanced), normal-size text


# --- colour math --------------------------------------------------------- #
def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _over(fg, alpha, bg):
    return tuple(fg[i] * alpha + bg[i] * (1 - alpha) for i in range(3))


def _lum(c):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(x) for x in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(fg, bg):
    a, b = _lum(fg), _lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _theme_vars(css, selector):
    """Extract `--name: #hex;` tokens from one rule block."""
    block = css[css.index(selector) + len(selector):]
    block = block[block.index("{") + 1: block.index("}")]
    out = {}
    for name, val in re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", block):
        out[name] = _rgb(val)
    return out


def _css():
    r = client.get("/static/styles.css")
    assert r.status_code == 200
    return r.text


def _html():
    return client.get("/").text


# --- 1.4.6 Contrast (Enhanced) ------------------------------------------- #
def test_status_and_text_colors_meet_aaa_7to1_both_themes():
    css = _css()
    for selector in (":root", '[data-theme="light"]'):
        v = _theme_vars(css, selector)
        panel, panel2 = v["panel"], v["panel-2"]
        # Each pair is (foreground, background) as actually composited in the UI.
        pairs = {
            "body text on panel": (v["text"], panel),
            "body text on panel-2": (v["text"], panel2),
            "muted on panel": (v["muted"], panel),
            "muted on panel-2": (v["muted"], panel2),
            "link on panel": (v["link"], panel),
            "link on panel-2": (v["link"], panel2),
            # status chips: the ink sits on a translucent tint of --ok/--err.
            "badge.ok ink": (v["ok-ink"], _over(v["ok"], 0.16, panel)),
            "badge.warn ink": (v["err-ink"], _over(v["err"], 0.16, panel)),
            "key-info.ok ink": (v["ok-ink"], _over(v["ok"], 0.12, panel)),
            "key-info.warn ink": (v["err-ink"], _over(v["err"], 0.12, panel)),
            "result .err ink": (v["err-ink"], panel2),
        }
        for label, (fg, bg) in pairs.items():
            r = _ratio(fg, bg)
            assert r >= AAA_NORMAL, f"{selector} {label}: {r:.2f}:1 < {AAA_NORMAL}:1"


# --- 2.5.5 Target Size (Enhanced) ---------------------------------------- #
def test_interactive_targets_are_at_least_44px():
    css = _css()
    # Every interactive control reserves a >=44px hit area.
    def block(selector):
        i = css.index(selector)
        return css[i: css.index("}", i)]
    for selector in (".btn {", ".theme-seg {", ".staged .rm {", ".skip-link {",
                     ".disclosure {", ".advanced summary"):
        assert "min-height: 44px" in block(selector), f"{selector} missing 44px target"
    # Form fields too.
    assert "min-height: 44px" in block("input[type=text]")


# --- 3.1.4 Abbreviations ------------------------------------------------- #
def test_domain_abbreviations_are_expanded():
    html = _html()
    # First use of each jargon abbreviation carries an expansion mechanism.
    for term in ("hOCR", "OCR", "API", "PDF", "TIFF", "JPEG", "DPI"):
        assert f">{term}</abbr>" in html, f"{term} not wrapped in <abbr>"
    assert '<abbr title="' in html


# --- 1.3.6 Identify Purpose (landmarks) ---------------------------------- #
def test_page_has_explicit_landmark_roles():
    html = _html()
    assert 'role="banner"' in html        # header
    assert 'role="contentinfo"' in html   # footer
    assert "<main" in html                 # main landmark


# --- 3.1.3 / 3.1.5 plain-language support -------------------------------- #
def test_plain_language_glossary_present():
    html = _html()
    assert 'class="glossary"' in html
    assert "Glossary" in html
    # Defines the core terms a newcomer wouldn't know.
    for term in ("Handwriting transcription", "Bounding box", "API key", "Token"):
        assert f"<dt>{term}</dt>" in html or f">{term}</dt>" in html
