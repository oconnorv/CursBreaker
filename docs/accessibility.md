# Accessibility conformance

CursBreaker targets **WCAG 2.1 Level AA in full**, and **Level AAA everywhere it
is achievable** for this kind of application. This note records what that means
in practice and the one AAA criterion that cannot be fully met.

## Level AAA — met

| Criterion | How it's met |
|---|---|
| **1.4.6 Contrast (Enhanced)** | Every text/UI colour pair clears **7:1** (4.5:1 for large text) in both the dark and light themes. Verified by computing the ratios from the actual CSS tokens in `tests/test_accessibility.py`. |
| **1.4.9 Images of Text (No Exception)** | No images of text — the wordmark and icons are styled text/glyphs. |
| **2.1.3 Keyboard (No Exception)** | All functionality uses native controls; full keyboard operation, visible focus ring, skip link. |
| **2.2.3 No Timing** | No time limits on interaction. (The localhost server's idle auto-shutdown is a process-lifecycle safeguard, kept alive while the tab is open; it is not a limit on reading or completing a task.) |
| **2.3.3 Animation from Interactions** | All transitions are disabled under `prefers-reduced-motion`. |
| **2.4.10 Section Headings** | Content is organised under real `h1`/`h2`/`h3` headings and landmark regions. |
| **2.5.5 Target Size (Enhanced)** | All pointer targets are ≥ **44×44 px** (buttons, the remove ×, theme toggle, form fields, disclosure summaries). Native radios/checkboxes use the user-agent-control exception. |
| **3.1.4 Abbreviations** | First use of each domain abbreviation (hOCR, ALTO, OCR, API, PDF, TIFF, JPEG, PNG, GIF, DPI, AGPL) is wrapped in `<abbr title>`, plus a page-level glossary. |
| **1.4.8 Visual Presentation** | User-selectable light/dark themes; body text ≤ ~80 characters; line spacing ≥ 1.5 in prose; no full-justification. |

Also reinforced beyond the letter of the spec: status is never conveyed by
colour alone (every state carries text **and** an icon/shape), which keeps the
ok/warn signalling legible under red-green colour blindness and in forced-colors
mode.

**Not applicable:** 1.2.6–1.2.9 and 1.4.7 (no audio or video), 2.4.8 Location
(single page), 2.2.5 Re-authenticating (no login).

## Level AAA — the known exception

**3.1.5 Reading Level.** The Success Criterion asks that text not require reading
ability beyond lower-secondary education, or that a simplified alternative be
provided. CursBreaker is inherently about specialist concepts — handwriting-text
recognition, hOCR/ALTO word geometry, bounding boxes, API tokens — so the core
material sits above that level and can't be rewritten without losing meaning.

Mitigations rather than removal:

- a **plain-language glossary** on the page defining the domain terms in everyday
  words;
- `<abbr>` expansions on first use; and
- plain-language helper text throughout (e.g. the "What is an API key?" walkthrough).

This is consistent with WCAG's own guidance that **Level AAA conformance is not
recommended as a blanket requirement for whole sites**, because some criteria
(reading level chief among them) cannot be satisfied for all content.

## Verifying

- Contrast (1.4.6) and the structural AAA work (target size, abbreviations,
  landmarks, glossary) are guarded by `tests/test_accessibility.py` — a colour
  change that drops below 7:1 fails the suite.
- Run `python -m pytest tests/test_accessibility.py`.
