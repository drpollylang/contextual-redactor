# ---- Start server using the following command ----
# uvicorn redactor.fastapi_app:app --reload --port 8008

# Mini FastAPI app for PDF text removal and reflow
# Duplicate email removal and document reflow logic
import base64
import io
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
import traceback
import sys



# -----------------------------
# Pydantic models
# -----------------------------
class Rect(BaseModel):
    x0: float = Field(..., description="Left X in page coordinates (points)")
    y0: float = Field(..., description="Top Y in page coordinates (points)")
    x1: float = Field(..., description="Right X in page coordinates (points)")
    y1: float = Field(..., description="Bottom Y in page coordinates (points)")

    def to_fitz(self) -> fitz.Rect:
        return fitz.Rect(self.x0, self.y0, self.x1, self.y1)


class Operation(BaseModel):
    page_index: int = Field(..., ge=0)
    rects: List[Rect]


class EditRequest(BaseModel):
    pdf_b64: str = Field(..., description="Original PDF as base64 string")
    operations: List[Operation]
    font_mode: Literal["auto", "base14", "helvetica"] = "auto"


class EditResponse(BaseModel):
    pdf_b64: str


# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI(title="PDF Remove & Reflow", version="1.0.0")

# Allow local dev origins (adjust for your environment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Font mapping helpers
# -----------------------------
BASE14 = {
    "Helvetica": "Helvetica",
    "Helvetica-Bold": "Helvetica-Bold",
    "Helvetica-Oblique": "Helvetica-Oblique",
    "Helvetica-BoldOblique": "Helvetica-BoldOblique",
    "Times-Roman": "Times-Roman",
    "Times-Bold": "Times-Bold",
    "Times-Italic": "Times-Italic",
    "Times-BoldItalic": "Times-BoldItalic",
    "Courier": "Courier",
    "Courier-Bold": "Courier-Bold",
    "Courier-Oblique": "Courier-Oblique",
    "Courier-BoldOblique": "Courier-BoldOblique",
    "Symbol": "Symbol",
    "ZapfDingbats": "ZapfDingbats",
}

def map_font(span_font: str, mode: str) -> str:
    """
    Best-effort font mapping for redraw.
    - auto: try original if Base-14 else Helvetica
    - base14: if span matches Base-14 use it else Helvetica
    - helvetica: force Helvetica
    """
    if mode == "helvetica":
        return "Helvetica"
    if mode == "base14":
        return span_font if span_font in BASE14 else "Helvetica"
    # auto
    return span_font if span_font in BASE14 else "Helvetica"


# -----------------------------
# Core algorithms
# -----------------------------
def get_spans_with_style(page: fitz.Page):
    """Extract spans with bbox, text, font, size, color in reading order."""
    d = page.get_text("dict")
    spans = []
    for b in d.get("blocks", []):
        if "lines" not in b:
            continue
        for ln in b["lines"]:
            for sp in ln["spans"]:
                spans.append({
                    "text": sp.get("text", ""),
                    "size": sp.get("size", 11),
                    "font": sp.get("font", "Helvetica"),
                    "color": sp.get("color", 0),  # 0xRRGGBB
                    "bbox": sp.get("bbox", [0, 0, 0, 0]),
                })
    return spans


# def remove_and_reflow_one_rect(page: fitz.Page, remove_rect: fitz.Rect, font_mode: str = "auto"):
#     """
#     Remove text spans intersecting 'remove_rect', shift spans below upwards by rect.height,
#     and redraw text. Images/drawings are not reflowed and will be dropped on this page.
#     """
#     spans = get_spans_with_style(page)
#     rh = remove_rect.height

#     # We will rebuild text => clear page contents first (this removes drawings/images too)
#     page.clean_contents()

#     tw = fitz.TextWriter(page.rect)

#     for sp in spans:
#         x0, y0, x1, y1 = sp["bbox"]
#         sp_rect = fitz.Rect(x0, y0, x1, y1)

#         # Skip spans that intersect the removal area
#         if sp_rect.intersects(remove_rect):
#             continue

#         # Shift spans that are below the removed block
#         dy = rh if sp_rect.y0 >= remove_rect.y1 else 0.0
#         baseline = y1 - dy  # TextWriter's Y is baseline

#         # fontname = map_font(sp["font"], font_mode)
#         # fontsize = sp["size"]
#         # rgb = ((sp["color"] >> 16) & 255, (sp["color"] >> 8) & 255, sp["color"] & 255)

#         # # Append text at the original X (no horizontal re-layout), new baseline Y
#         # tw.append(fitz.Point(x0, baseline), sp["text"], fontsize=fontsize, fontname=fontname, color=rgb)
        
#         font = get_font(page.parent, font_mode, sp["font"])
#         fontsize = sp["size"]
#         rgb = ((sp["color"] >> 16) & 255, (sp["color"] >> 8) & 255, sp["color"] & 255)

#         # TextWriter API in PyMuPDF >=1.23:
#         # append(point, text, font, fontsize=None, color=None, opacity=1, render_mode=0)
#         tw.append(
#             fitz.Point(x0, baseline),
#             sp["text"],
#             font,                # <---- Font object, NOT string
#             fontsize=fontsize,
#             color=rgb
#         )

#     tw.write_text(page)


# def remove_and_reflow_one_rect(page: fitz.Page, remove_rect: fitz.Rect, font_mode: str = "auto"):
#     print("PAGE SIZE:", page.rect)
#     print("REMOVE RECT:", remove_rect)
#     spans = get_spans_with_style(page)
#     rh = remove_rect.height

#     # clear text on the page
#     page.clean_contents()

#     tw = fitz.TextWriter(page.rect)

#     font_cache = {}

#     def get_font(span_font_name):
#         mapped = map_font(span_font_name, font_mode)
#         if mapped in font_cache:
#             return font_cache[mapped]
#         try:
#             f = fitz.Font(mapped)
#         except Exception:
#             f = fitz.Font("Helvetica")
#         font_cache[mapped] = f
#         return f

#     for sp in spans:
#         bbox = sp["bbox"]
#         if not bbox or len(bbox) != 4:
#             continue  # skip bad spans

#         x0, y0, x1, y1 = bbox
#         sp_rect = fitz.Rect(x0, y0, x1, y1)

#         # Skip spans inside removal rectangle
#         if sp_rect.intersects(remove_rect):
#             continue

#         # Shift downward content up
#         dy = rh if sp_rect.y0 >= remove_rect.y1 else 0.0
#         baseline = y1 - dy

#         font = get_font(sp["font"])
#         fontsize = sp["size"]
#         rgb = (
#             (sp["color"] >> 16) & 255,
#             (sp["color"] >> 8) & 255,
#             sp["color"] & 255,
#         )

#         tw.append(
#             fitz.Point(x0, baseline),
#             sp["text"],
#             font,
#             fontsize=fontsize,
#             # color=rgb
#         )

#     tw.write_text(page)



def remove_and_reflow_one_rect(page: fitz.Page, remove_rect: fitz.Rect, font_mode: str = "auto"):
    """
    Remove spans that intersect remove_rect and shift spans below upward by the vertical
    coverage of the *actually removed spans*. If no spans intersect, do nothing.
    """
    spans = get_spans_with_style(page)

    # Classify spans
    eps = 0.5  # small tolerance for float fuzz
    above, hit, below = [], [], []

    for sp in spans:
        bbox = sp.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        sp_rect = fitz.Rect(x0, y0, x1, y1)

        # Use intersects() (both x and y), but allow tiny tolerances
        if sp_rect.intersects(remove_rect):
            hit.append(sp)
        elif sp_rect.y1 <= remove_rect.y0 + eps:
            above.append(sp)
        elif sp_rect.y0 >= remove_rect.y1 - eps:
            below.append(sp)
        else:
            # Rare case: vertical overlap without horizontal (or vice versa). Treat as above/below by y.
            if sp_rect.y1 < remove_rect.y0:
                above.append(sp)
            elif sp_rect.y0 > remove_rect.y1:
                below.append(sp)
            else:
                # If in doubt, consider it hit so we don't leave artifacts
                hit.append(sp)

    # If nothing was actually removed, bail out without shifting
    if not hit:
        # No content intersected; page is left untouched
        return

    # Compute the vertical coverage of removed spans: min(y0) .. max(y1)
    min_y = min(sp["bbox"][1] for sp in hit)
    max_y = max(sp["bbox"][3] for sp in hit)
    removed_height = max(0.0, max_y - min_y)

    # Safety: if removed_height is suspiciously small, fallback to the rect height
    if removed_height < 0.5:
        removed_height = remove_rect.height

    # Clear and rebuild text
    page.clean_contents()
    tw = fitz.TextWriter(page.rect)

    font_cache = {}

    def get_font(span_font_name):
        mapped = map_font(span_font_name, font_mode)
        if mapped in font_cache:
            return font_cache[mapped]
        try:
            f = fitz.Font(mapped)  # PyMuPDF >= 1.23 uses Font objects
        except Exception:
            f = fitz.Font("Helvetica")
        font_cache[mapped] = f
        return f

    def write_span(sp, shift_y=0.0):
        x0, y0, x1, y1 = sp["bbox"]
        baseline = (y1 - shift_y)
        font = get_font(sp["font"])
        fontsize = sp["size"]
        rgb = ((sp["color"] >> 16) & 255, (sp["color"] >> 8) & 255, sp["color"] & 255)
        tw.append(fitz.Point(x0, baseline), sp["text"], font, fontsize=fontsize)

    # Write in reading order: all above (no shift) then below (shift)
    for sp in above:
        write_span(sp, shift_y=0.0)
    for sp in below:
        write_span(sp, shift_y=removed_height)

    tw.write_text(page)

    # (Optional) debug
    # print(f"[Reflow] page={page.number} removed_spans={len(hit)} shift={removed_height:.2f}")



# --- ADD / REPLACE in fastapi_app.py ---

def get_lines_with_style(page: fitz.Page):
    """
    Returns a list of "lines", each carrying:
      - bbox: (x0, y0, x1, y1) computed as union of its spans
      - spans: list of spans with bbox, text, font, size, color
    Preserves reading order from page.get_text('dict').
    """
    d = page.get_text("dict")
    lines = []
    for b in d.get("blocks", []):
        if "lines" not in b:
            continue
        for ln in b["lines"]:
            # union bbox of spans in this line
            span_objs = []
            min_x0 = 1e9; min_y0 = 1e9
            max_x1 = -1e9; max_y1 = -1e9
            for sp in ln.get("spans", []):
                bbox = sp.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = bbox
                min_x0 = min(min_x0, x0)
                min_y0 = min(min_y0, y0)
                max_x1 = max(max_x1, x1)
                max_y1 = max(max_y1, y1)
                span_objs.append({
                    "text": sp.get("text", ""),
                    "size": sp.get("size", 11),
                    "font": sp.get("font", "Helvetica"),
                    "color": sp.get("color", 0),
                    "bbox": [x0, y0, x1, y1],
                })
            if span_objs:
                lines.append({
                    "bbox": [min_x0, min_y0, max_x1, max_y1],
                    "spans": span_objs
                })
    return lines


def remove_and_reflow_by_lines(page: fitz.Page, remove_rect: fitz.Rect, font_mode: str = "auto", pad: float = 1.5):
    """
    Remove entire text lines whose vertical range overlaps 'remove_rect' (with tolerance),
    then shift all lines *below* the removed region upward by the *actual height* of the
    removed lines. If no lines intersect, do nothing.
    """
    lines = get_lines_with_style(page)
    if not lines:
        return  # nothing to do

    # Tolerance to account for tiny metric mismatches
    y0_t = remove_rect.y0 - pad
    y1_t = remove_rect.y1 + pad

    above, hit, below = [], [], []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]

        # Consider a line a "hit" when there is ANY vertical overlap
        vert_overlap = not (y1 < y0_t or y0 > y1_t)
        if vert_overlap:
            hit.append(ln)
        elif y1 <= y0_t:
            above.append(ln)
        else:
            below.append(ln)

    if not hit:
        # no lines removed -> bail without any shift to avoid distortion
        return

    # Actual removed height = vertical coverage of all hit lines
    min_y = min(ln["bbox"][1] for ln in hit)
    max_y = max(ln["bbox"][3] for ln in hit)
    removed_height = max(0.0, max_y - min_y)
    if removed_height < 0.5:
        removed_height = remove_rect.height  # fallback if metrics are super tight

    # Clear the page and redraw only text (like before)
    page.clean_contents()
    tw = fitz.TextWriter(page.rect)

    # Font cache (PyMuPDF >= 1.23)
    font_cache = {}
    def get_font(span_font_name):
        mapped = map_font(span_font_name, font_mode)
        if mapped in font_cache:
            return font_cache[mapped]
        try:
            f = fitz.Font(mapped)
        except Exception:
            f = fitz.Font("Helvetica")
        font_cache[mapped] = f
        return f

    def write_span(sp, shift_y=0.0):
        x0, y0, x1, y1 = sp["bbox"]
        baseline = (y1 - shift_y)
        font = get_font(sp["font"])
        fontsize = sp["size"]
        rgb = ((sp["color"] >> 16) & 255, (sp["color"] >> 8) & 255, sp["color"] & 255)
        tw.append(fitz.Point(x0, baseline), sp["text"], font, fontsize=fontsize)

    # Redraw above lines unchanged
    for ln in above:
        for sp in ln["spans"]:
            write_span(sp, shift_y=0.0)
    # Skip hit lines (removed)
    # Redraw below lines shifted upward by removed_height
    for ln in below:
        for sp in ln["spans"]:
            write_span(sp, shift_y=removed_height)

    tw.write_text(page)



@app.post("/v1/edit/remove-and-reflow", response_model=EditResponse)
def remove_and_reflow(req: EditRequest):
    try:
        try:
            pdf_bytes = base64.b64decode(req.pdf_b64)
        except Exception:
            print("ERROR: Invalid base64 in pdf_b64", file=sys.stderr)
            traceback.print_exc()
            raise HTTPException(status_code=400, detail="Invalid base64 in pdf_b64")

        try:
            doc = fitz.open("pdf", pdf_bytes)
        except Exception as e:
            print("ERROR: Cannot open PDF", file=sys.stderr)
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Unable to open PDF: {e}")

        try:
            apply_operations(doc, req.operations, req.font_mode)
        except Exception:
            print("ERROR: apply_operations crashed", file=sys.stderr)
            traceback.print_exc()
            raise

        out = doc.tobytes()
        doc.close()

        return EditResponse(pdf_b64=base64.b64encode(out).decode("utf-8"))

    except Exception:
        print("\n=== UNHANDLED SERVER ERROR ===", file=sys.stderr)
        traceback.print_exc()   # <--- FORCED TRACEBACK PRINT
        print("=== END ERROR ===\n", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal server error")



def _build_replacement_page(doc: fitz.Document,
                            width: float, height: float,
                            above_lines, below_lines,
                            removed_height: float,
                            font_mode: str) -> fitz.Document:
    """
    Create a temporary 1-page PDF containing only the redrawn text:
      - lines above: unchanged
      - lines below: shifted upward by removed_height
    Returns a new in-memory Document to be inserted into 'doc'.
    """
    tmp = fitz.open()
    newp = tmp.new_page(width=width, height=height)

    tw = fitz.TextWriter(fitz.Rect(0, 0, width, height))

    font_cache = {}
    def get_font(span_font_name: str):
        mapped = map_font(span_font_name, font_mode)
        if mapped in font_cache:
            return font_cache[mapped]
        try:
            f = fitz.Font(mapped)
        except Exception:
            f = fitz.Font("Helvetica")
        font_cache[mapped] = f
        return f

    def write_span(sp, shift_y=0.0):
        x0, y0, x1, y1 = sp["bbox"]
        baseline = (y1 - shift_y)
        font = get_font(sp["font"])
        fontsize = sp["size"]
        rgb = ((sp["color"] >> 16) & 255, (sp["color"] >> 8) & 255, sp["color"] & 255)
        tw.append(fitz.Point(x0, baseline), sp["text"], font, fontsize=fontsize)

    # Redraw content (text only)
    for ln in above_lines:
        for sp in ln["spans"]:
            write_span(sp, shift_y=0.0)
    for ln in below_lines:
        for sp in ln["spans"]:
            write_span(sp, shift_y=removed_height)

    tw.write_text(newp)
    return tmp


def remove_and_reflow_by_lines_inplace(doc: fitz.Document, page_index: int,
                                       remove_rect: fitz.Rect, font_mode: str = "auto",
                                       pad: float = 1.5) -> None:
    """
    Line-based removal + reflow that replaces the original page with a rebuilt one.
    This avoids overlay / ghosting because we never draw onto the original page.
    """
    if page_index < 0 or page_index >= len(doc):
        raise HTTPException(status_code=400, detail=f"Invalid page_index: {page_index}")

    page = doc[page_index]
    lines = get_lines_with_style(page)
    if not lines:
        return

    # Classify lines using vertical overlap with a small tolerance
    y0_t = remove_rect.y0 - pad
    y1_t = remove_rect.y1 + pad
    above, hit, below = [], [], []
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        vert_overlap = not (y1 < y0_t or y0 > y1_t)
        if vert_overlap:
            hit.append(ln)
        elif y1 <= y0_t:
            above.append(ln)
        else:
            below.append(ln)

    if not hit:
        # No lines intersect: do nothing (prevents spurious shifts / distortion)
        return

    # Compute actual vertical coverage of removed lines
    min_y = min(ln["bbox"][1] for ln in hit)
    max_y = max(ln["bbox"][3] for ln in hit)
    removed_height = max(0.0, max_y - min_y)
    if removed_height < 0.5:
        removed_height = remove_rect.height  # conservative fallback

    w, h = page.rect.width, page.rect.height

    # Build replacement page content in a temporary 1-page doc
    tmp = _build_replacement_page(doc, w, h, above, below, removed_height, font_mode)

    # Insert the rebuilt page *after* the current page,
    # then delete the original page to avoid index confusion.
    insert_at = page_index + 1
    doc.insert_pdf(tmp, from_page=0, to_page=0, start_at=insert_at)
    tmp.close()

    # Delete original page
    doc.delete_page(page_index)

    # After deletion, the newly inserted page shifts into 'page_index'
    # (We return to caller; if there are more rects on this page,
    #  the caller must re-fetch `page = doc[page_index]` before next op.)
    return


def apply_operations(doc: fitz.Document, ops: List[Operation], font_mode: str):
    """
    Apply rectangles per page, top->bottom. For pages with multiple rects,
    we must re-fetch the page after each replacement because page objects change.
    """
    # group by page
    by_page = {}
    for op in ops:
        by_page.setdefault(op.page_index, []).extend([r.to_fitz() for r in op.rects])

    for page_index, rects in by_page.items():
        # Sort rects top->bottom (y0 ascending)
        rects_sorted = sorted(rects, key=lambda r: r.y0)

        # IMPORTANT: after each in-place replacement, the page object changes,
        # but the page_index stays the same. So always operate with page_index
        # and re-fetch content each time in remove_and_reflow_by_lines_inplace.
        for r in rects_sorted:
            remove_and_reflow_by_lines_inplace(doc, page_index, r, font_mode=font_mode, pad=1.5)



# def apply_operations(doc: fitz.Document, ops: List[Operation], font_mode: str):
#     """
#     Apply multiple operations across pages. Within each page, apply rectangles
#     top->bottom so vertical shifts remain correct for subsequent rectangles.
#     """
#     # Group rects by page
#     by_page = {}
#     for op in ops:
#         by_page.setdefault(op.page_index, []).extend([r.to_fitz() for r in op.rects])

#     for page_index, rects in by_page.items():
#         if page_index < 0 or page_index >= len(doc):
#             raise HTTPException(status_code=400, detail=f"Invalid page_index: {page_index}")
#         page = doc[page_index]
#         # Sort by y0 (top to bottom)
#         rects_sorted = sorted(rects, key=lambda r: r.y0)
#         for r in rects_sorted:
#             #remove_and_reflow_one_rect(page, r, font_mode=font_mode)
#             remove_and_reflow_by_lines(page, r, font_mode=font_mode, pad=1.5)


# -----------------------------
# REST endpoint
# -----------------------------
@app.post("/v1/edit/remove-and-reflow", response_model=EditResponse)
def remove_and_reflow(req: EditRequest):
    try:
        pdf_bytes = base64.b64decode(req.pdf_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 in pdf_b64")

    try:
        doc = fitz.open("pdf", pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to open PDF: {e}")

    try:
        apply_operations(doc, req.operations, req.font_mode)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edit failed: {e}")

    out = doc.tobytes()
    doc.close()
    return EditResponse(pdf_b64=base64.b64encode(out).decode("utf-8"))
