
import fitz  # PyMuPDF
import re
import unicodedata
import math
from typing import List, Tuple, Pattern, Dict


# ===============================================================
# UTILITIES
# ===============================================================

def valid_rect(r: fitz.Rect) -> bool:
    """Return True only if rect is non-empty and coordinates finite."""
    return (
        r.x0 < r.x1 and
        r.y0 < r.y1 and
        all(math.isfinite(v) for v in [r.x0, r.y0, r.x1, r.y1])
    )


ZW = "".join([
    "\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF"
])
ZW_RE = re.compile(f"[{re.escape(ZW)}]")
NBSP_RE = re.compile("\u00A0")


def normalize_pdf_text(t: str) -> str:
    """Robust normalization for PDF-line matching."""
    t = unicodedata.normalize("NFKC", t)
    t = ZW_RE.sub("", t)
    t = NBSP_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ===============================================================
# EXTRACT LINES (supports spans AND chars)
# ===============================================================

def extract_lines(page: fitz.Page) -> List[Dict]:
    out = []
    raw = page.get_text("rawdict")

    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            line_parts = []
            y_tops = []
            y_bottoms = []
            baselines = []

            spans = ln.get("spans", [])
            if spans:
                for sp in spans:
                    # Case 1: direct span text
                    if "text" in sp and sp["text"]:
                        line_parts.append(sp["text"])
                        x0, y0, x1, y1 = sp["bbox"]
                        y_tops.append(y0)
                        y_bottoms.append(y1)
                        baselines.append(sp["origin"][1])

                    # Case 2: chars inside span
                    elif "chars" in sp and sp["chars"]:
                        chars = sp["chars"]
                        txt = "".join(ch.get("c", "") for ch in chars)
                        if txt:
                            line_parts.append(txt)
                        for ch in chars:
                            _, y0c, _, y1c = ch["bbox"]
                            y_tops.append(y0c)
                            y_bottoms.append(y1c)
                            baselines.append(ch["origin"][1])

            else:
                # Case 3: line-level chars
                chars = ln.get("chars", [])
                if chars:
                    txt = "".join(ch.get("c", "") for ch in chars)
                    if txt:
                        line_parts.append(txt)
                    for ch in chars:
                        _, y0c, _, y1c = ch["bbox"]
                        y_tops.append(y0c)
                        y_bottoms.append(y1c)
                        baselines.append(ch["origin"][1])
                
            text = "".join(line_parts).strip()
            height = max(y_bottoms) - min(y_tops)

            # Synthetic / ghost line filter
            if text == "" or height < 2.0:
                continue


            out.append({
                "text": "".join(line_parts),
                "y_top": min(y_tops),
                "y_bottom": max(y_bottoms),
                "baseline": max(baselines),
            })

    out.sort(key=lambda d: d["baseline"])
    return out


# ===============================================================
# GROUP CONSECUTIVE MATCHES
# ===============================================================

def group_consecutive(idxs: List[int]) -> List[Tuple[int, int]]:
    if not idxs:
        return []
    idxs.sort()
    groups = []
    s = prev = idxs[0]
    for i in idxs[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append((s, prev))
            s = prev = i
    groups.append((s, prev))
    return groups


# ===============================================================
# SPILL CONTENT INTO PREVIOUS PAGE
# ===============================================================

def spill_to_previous_page(
    doc: fitz.Document,
    page_index: int,
    overflow_top: float,
    overflow_bottom: float,
    debug=False,
) -> bool:
    if page_index <= 0:
        return False

    src = doc[page_index]
    prev = doc[page_index - 1]
    height = overflow_bottom - overflow_top

    if height <= 0:
        return False

    # Find bottom-most Y on previous page
    raw = prev.get_text("rawdict")
    prev_bottom = 0
    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                prev_bottom = max(prev_bottom, sp["bbox"][3])
            for c in ln.get("chars", []):
                prev_bottom = max(prev_bottom, c["bbox"][3])

    available = prev.rect.height - prev_bottom - 5
    if available < height:
        return False

    # Create source clone
    temp = fitz.open()
    temp.insert_pdf(doc, from_page=page_index, to_page=page_index)

    target_rect = fitz.Rect(0, prev_bottom + 2,
                            src.rect.width, prev_bottom + 2 + height)
    clip_rect = fitz.Rect(0, overflow_top,
                          src.rect.width, overflow_bottom)

    if valid_rect(target_rect) and valid_rect(clip_rect):
        prev.show_pdf_page(target_rect, temp, 0, clip=clip_rect)
        if debug:
            print(f"  [spill] moved {height:.2f} pts from p{page_index+1} → p{page_index}")
        return True

    return False




def remove_page_separators(page):
    drawings = page.get_drawings()
    for d in drawings:
        rect = d["rect"]
        # Identify horizontal bars (thin height, wide width)
        if rect.height < 4 and rect.width > page.rect.width * 0.5:
            # Erase the bar using white rectangle
            shape = page.new_shape()
            shape.draw_rect(rect, color=None, fill=(1,1,1))
            shape.commit()


# ===============================================================
# COLLAPSE GAP (with top-margin handling + spill support)
# ===============================================================

def collapse_gap(
    doc: fitz.Document,
    page_index: int,
    gap_top: float,
    gap_bottom: float,
    *,
    allow_spill=True,
    debug=False
):
    assert gap_bottom > gap_top

    page = doc[page_index]
    w, h = page.rect.width, page.rect.height
    shift = gap_bottom - gap_top

    # --- Determine true top margin ---
    raw = page.get_text("rawdict")
    tops = []
    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                tops.append(sp["bbox"][1])
            for c in ln.get("chars", []):
                tops.append(c["bbox"][1])
    page_top = min(tops) if tops else 0  # visual top margin (~72)

    # How far collapse pushes content upward
    new_gap_top = gap_top - shift

    # Overflow is anything above visual top
    overflow_height = max(0, page_top - new_gap_top)

    # --- SPILL TO PREVIOUS PAGE ---
    if allow_spill and overflow_height > 0:
        success = spill_to_previous_page(
            doc, page_index,
            overflow_top=new_gap_top,
            overflow_bottom=new_gap_top + overflow_height,
            debug=debug
        )
        if success:
            # Adjust local collapse region
            gap_top += overflow_height
            shift = gap_bottom - gap_top
            new_gap_top = gap_top - shift
            if debug:
                print(f"  [adjusted] gap_top={gap_top:.2f}, shift={shift:.2f}")

    # --- PERFORM PAGE COMPOSITING COLLAPSE ---
    temp = fitz.open()
    temp.insert_pdf(doc, from_page=page_index, to_page=page_index)

    composed = doc.new_page(page_index + 1, width=w, height=h)

    upper_clip = fitz.Rect(0, 0, w, gap_top)
    if valid_rect(upper_clip):
        composed.show_pdf_page(upper_clip, temp, 0, clip=upper_clip)

    lower_clip = fitz.Rect(0, gap_bottom, w, h)
    lower_target = fitz.Rect(0, gap_bottom - shift, w, h - shift)
    if valid_rect(lower_clip) and valid_rect(lower_target):
        composed.show_pdf_page(lower_target, temp, 0, clip=lower_clip)

    # Blank bottom region
    if shift > 0:
        shape = composed.new_shape()
        shape.draw_rect(fitz.Rect(0, h - shift, w, h))
        shape.commit()

    doc.delete_page(page_index)


# ===============================================================
# MAIN PIPELINE
# ===============================================================

def remove_matches_and_collapse(
    input_pdf: str,
    output_pdf: str,
    patterns: List[str],
    *,
    include_following_space=True,
    debug=False,
):
    regexes = [re.compile(p, re.IGNORECASE) for p in patterns]
    doc = fitz.open(input_pdf)

    for page_index in range(len(doc)):
        
        if debug:
            print(f"\n=== PAGE {page_index+1} ===")

        page = doc[page_index]
        lines = extract_lines(page)

        remove_page_separators(page)

                
        page.draw_rect(fitz.Rect(100, -100, 200, 200), fill=(1, 1, 1))


        # ---- STEP 0: REMOVE TOP MARGIN ----
        if lines:
            first_top = lines[0]["y_top"]
            if first_top > 1.0:
                if debug:
                    print(f"Removing top margin [0 → {first_top:.2f}]")
                collapse_gap(doc, page_index, 0, first_top, allow_spill=False, debug=debug)
                page = doc[page_index]
                lines = extract_lines(page)

        # ---- STEP 1: find matching lines ----
        matched = []
        for i, ln in enumerate(lines):
            txt = normalize_pdf_text(ln["text"])
            if any(r.search(txt) for r in regexes):
                matched.append(i)

        if not matched:
            if debug:
                print("No matches.")
            continue

        groups = group_consecutive(matched)
        if debug:
            print("Matched groups:", groups)

        total_shift = 0

        # ---- STEP 2: collapse each group ----
        for g0, g1 in groups:
            top = lines[g0]["y_top"]
            bottom = lines[g1]["y_bottom"]

            if include_following_space:
                nxt = g1 + 1
                if nxt < len(lines):
                    bottom = lines[nxt]["y_top"]

            y0 = top - total_shift
            y1 = bottom - total_shift

            if debug:
                print(f"collapsing [{y0:.2f} → {y1:.2f}] "
                      f"(orig {top:.2f} → {bottom:.2f})")

            collapse_gap(doc, page_index, y0, y1, allow_spill=True, debug=debug)

            total_shift += (bottom - top)

        if page_index == 0:    
            print([ (i, ln["y_top"], normalize_pdf_text(ln["text"]) ) for i, ln in enumerate(lines) ])


    doc.save(output_pdf)
    doc.close()


# ===============================================================
# Example Runner
# ===============================================================
if __name__ == "__main__":
    # COLON = r"[:\uFF1A\uFE55\uFE13\u2236]"
    # patterns = [
    #     rf"^From{COLON}\s*.*",
    #     rf"^To{COLON}\s*.*",
    #     rf"^CC{COLON}\s*.*",
    #     rf"^Subject{COLON}\s*.*",
    #     rf"^Date{COLON}\s*.*",
    # ]
    patterns = [r"^From:\s*(.+)", r"^To:\s*(.+)", r"^Subject:\s*(.+)", r"^Date:\s*(.+)"]

    remove_matches_and_collapse(
        "FakeStackedEmailThread.pdf",
        "Cleaned_FakeStackedEmailThread.pdf",
        patterns,
        debug=True,
    )
