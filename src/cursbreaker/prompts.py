"""Transcription prompts, shared by every provider.

These live apart from any one SDK client because all three providers (Gemini,
Claude, OpenAI) are asked for exactly the same thing: a faithful, line-by-line
transcription, and -- for the box-producing calls -- a ``box_2d`` per line on a
0-1000 normalized grid. Keeping one copy means a wording improvement lands for
every provider at once, and it keeps provider comparisons honest: a difference
in output is a difference in the model, not in what we asked for.

The wording follows Mark Humphries' "Gemini 3 Solves Handwriting Recognition"
recipe -- transcribe exactly, expand nothing, one source line per output line.
"""

from __future__ import annotations

_READING_ORDER_RULE = (
    "Use natural reading order: if the page has multiple columns, finish each "
    "column from top to bottom before moving on to the next column (left to "
    "right). For a single column, just go top to bottom."
)

PROMPT_TRANSCRIBE = (
    "You are an expert paleographer. Carefully transcribe the handwriting in "
    "this document image. Transcribe every line of the main text, preserving "
    "the original line breaks (one source line per output line). "
    + _READING_ORDER_RULE
    + " Expand nothing, correct nothing, and translate nothing - reproduce the "
    "text exactly as written. Respond with ONLY the transcription text: no "
    "commentary, labels, or code fences."
)

PROMPT_DETECT = (
    "Detect every line of handwritten or printed text in this document image. "
    "Return a JSON array where each element has two fields: 'text' (your "
    "accurate transcription of that single line, exactly as written) and "
    "'box_2d' (the line's bounding box as [ymin, xmin, ymax, xmax], integers "
    "normalized to 0-1000 with the origin at the top-left). "
    + _READING_ORDER_RULE
    + " One element per source line. Do not merge separate lines. Never return "
    "masks, explanations, or code fences."
)

PROMPT_ONE_PASS = (
    "Carefully transcribe the handwriting in this document image, line by line. "
    "Return a JSON array where each element has 'text' (the accurate "
    "transcription of one source line, reproduced exactly as written) and "
    "'box_2d' ([ymin, xmin, ymax, xmax] integers normalized to 0-1000, origin "
    "top-left). "
    + _READING_ORDER_RULE
    + " One element per source line. Never return masks, explanations, or code "
    "fences."
)

# Appended for providers whose structured-output mode returns a JSON *object*
# rather than a bare array (Claude and OpenAI both wrap the list in a field).
PROMPT_LINES_ENVELOPE = (
    " Return the array as the 'lines' field of a single JSON object."
)
