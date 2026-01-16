
import fitz  # PyMuPDF
import re
import unicodedata
from typing import List, Tuple, Pattern, Dict


# -------------------------------
# Unicode / whitespace normalization for robust matching
# -------------------------------
_ZW = "".join([
    "\u200B",  # ZERO WIDTH SPACE
    "\u200C",  # ZERO WIDTH NON-JOINER
    "\u200D",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
    "\uFEFF",  # ZERO WIDTH NO-BREAK SPACE
])
_ZW_RE = re.compile(f"[{re.escape(_ZW)}]")
_NBSP_RE = re.compile("\u00A0")  # NO-BREAK SPACE

def normalize_pdf_text(t: str) -> str:
    # Unicode normalization (fix ligatures / fancy punctuation / fullwidth forms)
    t = unicodedata.normalize("NFKC", t)
    # Remove zero-width & NBSP
    t = _ZW_RE.sub("", t)
    t = _NBSP_RE.sub(" ", t)
    # Collapse whitespace & trim
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -------------------------------
# Extract visual lines (span-accurate; supports spans with text OR chars)
# -------------------------------
def extract_lines(page: fitz.Page) -> List[Dict]:
    """
    Returns lines sorted top->bottom with:
      - text            : concatenated span/char text
      - y_top, y_bottom : visual vertical band of the line
      - baseline        : dominant baseline (for ordering)
    Works with rawdict where spans may have 'text' or 'chars'.
    """
    out: List[Dict] = []
    raw = page.get_text("rawdict")

    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            line_text_parts: List[str] = []
            y_tops: List[float] = []
            y_bottoms: List[float] = []
            baselines: List[float] = []

            spans = ln.get("spans", [])
            if spans:
                for sp in spans:
                    # Case 1: span has 'text'
                    if "text" in sp and sp["text"]:
                        txt = sp["text"]
                        line_text_parts.append(txt)
                        x0, y0, x1, y1 = sp["bbox"]
                        y_tops.append(y0); y_bottoms.append(y1)
                        baselines.append(sp["origin"][1])
                    # Case 2: span has 'chars' only
                    elif "chars" in sp and sp["chars"]:
                        chars = sp["chars"]
                        txt = "".join(ch.get("c", "") for ch in chars)
                        if txt:
                            line_text_parts.append(txt)
                        for ch in chars:
                            _, y0c, _, y1c = ch["bbox"]
                            y_tops.append(y0c); y_bottoms.append(y1c)
                            baselines.append(ch["origin"][1])
            else:
                # Some PDFs may provide line-level chars
                chars = ln.get("chars", [])
                if chars:
                    txt = "".join(ch.get("c", "") for ch in chars)
                    if txt:
                        line_text_parts.append(txt)
                    for ch in chars:
                        _, y0c, _, y1c = ch["bbox"]
                        y_tops.append(y0c); y_bottoms.append(y1c)
                        baselines.append(ch["origin"][1])

            if not line_text_parts or not y_tops:
                continue

            text = "".join(line_text_parts)
            y_top = min(y_tops)
            y_bottom = max(y_bottoms)
            baseline = max(baselines) if baselines else (y_bottom + y_top) / 2.0

            out.append({
                "text": text,
                "y_top": y_top,
                "y_bottom": y_bottom,
                "baseline": baseline,
            })

    out.sort(key=lambda d: d["baseline"])
    return out


# -------------------------------
# Compile regex patterns
# -------------------------------
def compile_patterns(patterns: List[str]) -> List[Pattern]:
    return [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]


# -------------------------------
# Utility: group consecutive matched line indices
# -------------------------------
def group_consecutive(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    indices.sort()
    groups = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append((start, prev))
            start = prev = i
    groups.append((start, prev))
    return groups


# -------------------------------
# Compositing: delete band and shift lower content up
# -------------------------------
def collapse_gap_by_compositing(
    doc: fitz.Document,
    page_index: int,
    gap_top: float,
    gap_bottom: float,
    *,
    fill_bottom: bool = True,
):
    assert gap_bottom > gap_top, "gap_bottom must be > gap_top"

    src = doc[page_index]
    w, h = src.rect.width, src.rect.height
    shift = gap_bottom - gap_top

    # Clone to temp doc (avoid source=target restriction)
    temp = fitz.open()
    temp.insert_pdf(doc, from_page=page_index, to_page=page_index)

    composed = doc.new_page(page_index + 1, width=w, height=h)

    # Upper unchanged
    upper_clip = fitz.Rect(0, 0, w, gap_top)
    composed.show_pdf_page(upper_clip, temp, 0, clip=upper_clip)

    # Lower shifted up
    lower_clip = fitz.Rect(0, gap_bottom, w, h)
    lower_target = fitz.Rect(0, gap_bottom - shift, w, h - shift)
    composed.show_pdf_page(lower_target, temp, 0, clip=lower_clip)

    # Fill vacated bottom strip
    if fill_bottom and shift > 0:
        shape = composed.new_shape()
        shape.draw_rect(fitz.Rect(0, h - shift, w, h))
        shape.commit()

    # Replace original page
    doc.delete_page(page_index)



def shift_overflow_to_previous_page(
    doc: fitz.Document,
    page_index: int,
    overflow_y0: float,  # typically 0
    overflow_y1: float,  # the part that would go above the top
):
    """
    Move the vertical slice [overflow_y0, overflow_y1) from page_index
    onto the bottom of page_index - 1, IF it fits.

    Returns True if moved; False if not moved.
    """
    if page_index <= 0:
        return False  # can't spill onto previous page

    src = doc[page_index]
    prev = doc[page_index - 1]

    w = src.rect.width
    h_prev = prev.rect.height

    # How tall is the overflow slice?
    height = overflow_y1 - overflow_y0
    if height <= 0:
        return False

    # Determine lowest occupied Y on prev page
    prev_raw = prev.get_text("rawdict")
    prev_bottom = 0
    for b in prev_raw.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                if "text" in sp:
                    prev_bottom = max(prev_bottom, sp["bbox"][3])
                elif "chars" in sp:
                    for c in sp["chars"]:
                        prev_bottom = max(prev_bottom, c["bbox"][3])

    available = h_prev - prev_bottom - 5  # 5pt padding

    if available < height:
        return False  # Not enough room

    # ---- Perform the transfer using compositing ----

    # 1. Clone the source page into a temp doc
    temp_src = fitz.open()
    temp_src.insert_pdf(doc, from_page=page_index, to_page=page_index)

    # 2. Draw the overflow slice onto the previous page
    target_rect = fitz.Rect(0, prev_bottom + 2, w, prev_bottom + 2 + height)
    overflow_clip = fitz.Rect(0, overflow_y0, w, overflow_y1)

    prev.show_pdf_page(target_rect, temp_src, 0, clip=overflow_clip)

    return True



def collapse_gap_with_spill(
    doc: fitz.Document,
    page_index: int,
    gap_top: float,
    gap_bottom: float,
    allow_spill: bool = True,
    fill_bottom: bool = True,
):
    """
    Collapse whitespace created by a removed block.
    If content would be pushed above the top of the page, optionally spill
    the overflow onto the previous page (if space allows).
    """
    assert gap_bottom > gap_top

    src = doc[page_index]
    w, h = src.rect.width, src.rect.height
    shift = gap_bottom - gap_top

    # Compute what region will overflow upward
    # new_gap_top = gap_top - shift
    # overflow_height = max(0, -new_gap_top)
    
    # Determine the page's true top margin by measuring the top-most text
    raw = doc[page_index].get_text("rawdict")
    tops = []
    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                tops.append(sp["bbox"][1])
            for c in ln.get("chars", []):
                tops.append(c["bbox"][1])
    PAGE_TOP = min(tops) if tops else 0  # ~72 pts in your case

    new_gap_top = gap_top - shift

    # Overflow is anything that tries to move above PAGE_TOP, *not y=0*
    overflow_height = max(0, PAGE_TOP - new_gap_top)


    # ---- A) Try to spill overflow upward into previous page ----
    if allow_spill and overflow_height > 0:
        moved = shift_overflow_to_previous_page(
            doc,
            page_index=page_index,
            overflow_y0=0,
            overflow_y1=overflow_height
        )
        if moved:
            # After moving overflow upward,
            # the remaining collapse is from [gap_top, gap_bottom)
            # BUT with no overflow now.
            gap_top = overflow_height
            shift = gap_bottom - gap_top

    # ---- B) Collapse the remaining gap on this page ----

    # Clone source page into temp doc (PyMuPDF requirement)
    temp = fitz.open()
    temp.insert_pdf(doc, from_page=page_index, to_page=page_index)

    composed = doc.new_page(page_index + 1, width=w, height=h)

    # Upper unchanged region
    upper_clip = fitz.Rect(0, 0, w, gap_top)
    composed.show_pdf_page(upper_clip, temp, 0, clip=upper_clip)

    # Lower shifted region
    lower_clip = fitz.Rect(0, gap_bottom, w, h)
    lower_target = fitz.Rect(0, gap_bottom - shift, w, h - shift)
    composed.show_pdf_page(lower_target, temp, 0, clip=lower_clip)

    # Fill bottom to avoid ghosting
    if fill_bottom and shift > 0:
        shape = composed.new_shape()
        shape.draw_rect(fitz.Rect(0, h - shift, w, h))
        shape.commit()

    # Replace page
    doc.delete_page(page_index)



# -------------------------------
# Main: regex match & partial collapse (Option A)
# -------------------------------
def remove_matches_and_collapse(
    input_pdf: str,
    output_pdf: str,
    patterns: List[str],
    *,
    include_following_space: bool = True,
    extra_pad_above: float = 0.0,
    extra_pad_below: float = 0.0,
    normalize_text: bool = True,
    debug_print: bool = False,
):
    """
    For each page:
      - Build visual lines from spans or chars.
      - Match lines by regex (after normalization if enabled).
      - Group consecutive matches into blocks.
      - For each block, delete from the top of the first matched line
        to the top of the next **non-matched** line (if include_following_space),
        or to the bottom of the last matched line otherwise.
      - Shift lower content up by exactly that height (layout preserved).
    """
    doc = fitz.open(input_pdf)
    regs = compile_patterns(patterns)

    for page_index in range(len(doc)):
        page = doc[page_index]
        lines = extract_lines(page)
        
        # STEP 0: Remove top margin so spill-up can work.
        # Collapse everything between y=0 and the first line's y_top

        first_line_y = lines[0]["y_top"] if lines else None
        if first_line_y and first_line_y > 1.0:
            collapse_gap_with_spill(
                doc, page_index,
                gap_top=0,
                gap_bottom=first_line_y,
                allow_spill=False  # margin collapse never spills upward
            )
            # After replacing page, re-extract lines
            page = doc[page_index]
            lines = extract_lines(page)

        if debug_print:
            print(f"[page {page_index+1}] extracted lines: {len(lines)}")
            for i, ln in enumerate(lines[:15]):
                print(f"  L{i:02d}:", normalize_pdf_text(ln['text']) if normalize_text else ln['text'])

        # Find matching lines
        matched_idxs: List[int] = []
        for i, ln in enumerate(lines):
            s = normalize_pdf_text(ln["text"]) if normalize_text else ln["text"]
            if any(r.search(s) for r in regs):
                matched_idxs.append(i)

        if not matched_idxs:
            if debug_print:
                print(f"[page {page_index+1}] no matches")
            continue

        groups = group_consecutive(matched_idxs)
        if debug_print:
            print(f"[page {page_index+1}] matched groups: {groups}")

        total_shift = 0.0
        for g0, g1 in groups:
            # Compute original band to remove
            band_top = lines[g0]["y_top"]
            band_bottom = lines[g1]["y_bottom"]

            if include_following_space:
                nxt = g1 + 1
                if nxt < len(lines):
                    # Expand removal to consume the gap up to the next non-matched line
                    band_bottom = max(band_bottom, lines[nxt]["y_top"])

            # Padding / clamp
            h = doc[page_index].rect.height
            band_top = max(0.0, band_top - extra_pad_above)
            band_bottom = min(h, band_bottom + extra_pad_below)

            # Adjust by cumulative shift from previous collapses on this page
            y0 = band_top - total_shift
            y1 = band_bottom - total_shift
            if y1 <= y0:
                continue

            if debug_print:
                print(f"[page {page_index+1}] collapse [{y0:.2f}, {y1:.2f}] "
                      f"(orig [{band_top:.2f}, {band_bottom:.2f}])")

            # collapse_gap_by_compositing(doc, page_index, y0, y1)
            collapse_gap_with_spill(doc, page_index, y0, y1, allow_spill=True)
            total_shift += (band_bottom - band_top)

    doc.save(output_pdf)
    doc.close()


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Robust header-like patterns. Feel free to add/remove.
    # We allow various colon-like characters: ":" "：" etc.
    COLON = r"[:\uFF1A\uFE55\uFE13\u2236]"
    # patterns = [
    #     rf"^From{COLON}\s*.*$",
    #     rf"^To{COLON}\s*.*$",
    #     rf"^CC{COLON}\s*.*$",
    #     rf"^Subject{COLON}\s*.*$",
    #     rf"^Date{COLON}\s*.*$",
    #     # Example of a typical “quoted reply” line:
    #     # r"^On \d{1,2} \w{3} \d{4}, at \d{2}:\d{2}, .+ wrote{COLON}$",
    # ]
    patterns = [r"^From:\s*(.+)", r"^To:\s*(.+)", r"^Subject:\s*(.+)", r"^Date:\s*(.+)"]

    remove_matches_and_collapse(
        input_pdf="FakeStackedEmailThread.pdf",
        output_pdf="Cleaned_FakeStackedEmailThread.pdf",
        patterns=patterns,
        include_following_space=True,   # eat the blank gap after the header block
        extra_pad_above=0.0,
        extra_pad_below=0.0,
        normalize_text=True,
        debug_print=True,               # turn on to verify matches and bands
    )
