
import fitz
from PIL import Image, ImageDraw
import io
import re
import unicodedata
import numpy as np


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize(t: str):
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------
# PDF → line extraction (dict)
# Used only to find header blocks
# ---------------------------------------------------------

def extract_header_bands(page, patterns):
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    matches = []

    d = page.get_text("dict")
    for blk in d.get("blocks", []):
        if "lines" not in blk:
            continue
        text = "".join(
            sp.get("text", "")
            for ln in blk["lines"]
            for sp in ln.get("spans", [])
            if sp.get("text")
        )
        if not text.strip():
            continue

        norm = normalize(text)
        if any(r.search(norm) for r in regs):
            x0, y0, x1, y1 = blk["bbox"]
            matches.append((y0, y1))

    if not matches:
        return None

    return min(y0 for y0, _ in matches), max(y1 for _, y1 in matches)


# ---------------------------------------------------------
# Rasterization
# ---------------------------------------------------------

def raster_page(page, zoom):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def mask_header(img, top_px, bot_px):
    draw = ImageDraw.Draw(img)
    W, H = img.size
    top_px = max(0, top_px)
    bot_px = min(H, bot_px)
    draw.rectangle([(0, top_px), (W, bot_px)], fill="white")
    return img


def trim_top_bottom(img, white_thresh=245):
    """Trim vertical whitespace at top and bottom."""
    gray = np.array(img.convert("L"))
    H, W = gray.shape

    # find first non-white
    top = 0
    for y in range(H):
        if np.mean(gray[y]) < white_thresh:
            top = y
            break

    # find last non-white
    bottom = H - 1
    for y in range(H - 1, -1, -1):
        if np.mean(gray[y]) < white_thresh:
            bottom = y
            break

    # If entirely white, return blank page
    if bottom <= top:
        return Image.new("RGB", img.size, "white")

    return img.crop((0, top, W, bottom + 1))


# ---------------------------------------------------------
# Continuous Scroll → Dynamic slicing into original page size
# ---------------------------------------------------------


def repaginate_scroll(giant_img, target_height, white_threshold=245, min_content_rows=20):
    """
    Slice giant scroll into pages of approx target_height,
    but avoid slicing inside large whitespace regions.
    Ensures pages start and end on meaningful content.
    """
    gray = np.array(giant_img.convert("L"))
    H, W = gray.shape

    pages = []
    cursor = 0

    while cursor < H:
        # Tentative slice
        tentative_end = min(cursor + target_height, H)
        slice_gray = gray[cursor:tentative_end]

        # Find visible content in this slice
        # Top content
        top = None
        for y in range(slice_gray.shape[0]):
            if np.mean(slice_gray[y]) < white_threshold:
                top = y
                break

        # If no content at all in this slice → skip it entirely
        if top is None:
            cursor = tentative_end
            continue

        # Bottom content
        bottom = None
        for y in range(slice_gray.shape[0] - 1, -1, -1):
            if np.mean(slice_gray[y]) < white_threshold:
                bottom = y
                break

        # Expand slice downwards if content does not fill the page
        # (avoid big whitespace blocks between logical paragraphs)
        if bottom - top < min_content_rows and tentative_end < H:
            # try to include content from below
            extension = int(target_height * 0.4)  # extra 40% downwards
            extend_to = min(tentative_end + extension, H)
            slice_gray2 = gray[cursor:extend_to]

            # recompute top/bottom in extended slice
            new_top = None
            for y in range(slice_gray2.shape[0]):
                if np.mean(slice_gray2[y]) < white_threshold:
                    new_top = y
                    break

            new_bottom = None
            for y in range(slice_gray2.shape[0]-1, -1, -1):
                if np.mean(slice_gray2[y]) < white_threshold:
                    new_bottom = y
                    break

            # Use the extended content if it's better
            if new_top is not None and new_bottom - new_top > bottom - top:
                top = new_top
                bottom = new_bottom
                tentative_end = cursor + slice_gray2.shape[0]

        # Now crop the slice to just the meaningful content range
        real_top = cursor + top
        real_bottom = cursor + bottom + 1

        page_img = giant_img.crop((0, real_top, W, real_bottom))
        pages.append(page_img)

        cursor = real_bottom

    return pages


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def continuous_reflow_email_pdf(
    input_pdf,
    output_pdf,
    patterns,
    *,
    zoom=2.0,
    white_threshold=245,
    debug=False
):
    src = fitz.open(input_pdf)

    raster_pages = []
    page_heights = []
    page_widths = []

    # 1. Rasterize + header mask + trim whitespace
    for i, page in enumerate(src):
        img = raster_page(page, zoom)
        W, H = img.size
        page_widths.append(W)
        page_heights.append(H)

        band = extract_header_bands(page, patterns)
        if band:
            y0, y1 = band
            img = mask_header(img, int(y0 * zoom), int(y1 * zoom))

        # Trim top and bottom whitespace
        img = trim_top_bottom(img, white_thresh=white_threshold)
        raster_pages.append(img)

    # 2. Build one continuous scroll image
    W = max(page_widths)
    total_H = sum(img.size[1] for img in raster_pages)

    if debug:
        print(f"Scroll size: W={W}, H={total_H}")

    giant = Image.new("RGB", (W, total_H), "white")
    y = 0
    for img in raster_pages:
        giant.paste(img, (0, y))
        y += img.size[1]

    # 3. Repaginate the giant scroll into original page size
    target_height = page_heights[0]  # match original PDF page height
    slices = repaginate_scroll(giant, target_height, white_threshold=white_threshold)

    # 4. Output final PDF
    out = fitz.open()
    for s in slices:
        w, h = s.size
        page = out.new_page(width=w, height=h)
        buf = io.BytesIO()
        s.save(buf, format="PNG")
        page.insert_image(page.rect, stream=buf.getvalue())

    out.save(output_pdf)
    out.close()


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

if __name__ == "__main__":
    # Allow variants of colon
    # COLON = r"[:\uFF1A\uFE55\uFE13\u2236]"

    # patterns = [
    #     rf"^From{COLON}\s*.*",
    #     rf"^To{COLON}\s*.*",
    #     rf"^CC{COLON}\s*.*",
    #     rf"^Subject{COLON}\s*.*",
    #     rf"^Date{COLON}\s*.*",
    # ]
    patterns = [r"^From:\s*(.+)", r"^To:\s*(.+)", r"^Subject:\s*(.+)", r"^Date:\s*(.+)"]

    continuous_reflow_email_pdf(
        "FakeStackedEmailThread.pdf",
        "Cleaned_EmailThread_Final.pdf",
        patterns,
        zoom=1.0,
        debug=True
    )
