
import fitz               # PyMuPDF
from PIL import Image
import io
import re
import unicodedata
import numpy as np

TOP_MARGIN_PX = 40   # slightly smaller than top margin (in pixels)

# ---------------------------
# Text normalization
# ---------------------------
def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# ---------------------------
# Line extraction (dict → lines)
# Returns sorted list of dicts:
#   { "text": str, "y_top": float, "y_bottom": float }
# ---------------------------
def extract_lines_dict(page) -> list:
    lines = []
    d = page.get_text("dict")
    for blk in d.get("blocks", []):
        for ln in blk.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            # Stitch span texts for the (visual) line
            text = "".join(sp.get("text", "") for sp in spans if sp.get("text"))
            text = text.strip()
            if text == "":
                continue
            # Line bbox
            x0s, y0s, x1s, y1s = [], [], [], []
            for sp in spans:
                bx0, by0, bx1, by1 = sp["bbox"]
                x0s.append(bx0); y0s.append(by0); x1s.append(bx1); y1s.append(by1)
            y_top = min(y0s); y_bottom = max(y1s)
            lines.append({"text": text, "y_top": y_top, "y_bottom": y_bottom})
    # top → bottom
    lines.sort(key=lambda L: L["y_top"])
    return lines

# ---------------------------
# Group consecutive indices
# ---------------------------
def group_consecutive(idxs):
    if not idxs:
        return []
    idxs = sorted(idxs)
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

# ---------------------------
# Image tools
# ---------------------------
def pil_from_pixmap(pix: fitz.Pixmap) -> Image.Image:
    return Image.open(io.BytesIO(pix.tobytes("png")))

def delete_rows_and_collapse(img: Image.Image, y0_px: int, y1_px: int) -> Image.Image:
    """
    Remove rows [y0_px:y1_px] and collapse upwards.
    """
    if y1_px <= y0_px:
        return img
    arr = np.array(img)              # H x W x C
    h, w, c = arr.shape
    y0_px = max(0, min(y0_px, h))
    y1_px = max(0, min(y1_px, h))
    if y1_px <= y0_px:
        return img
    # Stack top + bottom
    top = arr[:y0_px, :, :]
    bottom = arr[y1_px:, :, :]
    collapsed = np.vstack([top, bottom])
    # Fill remainder with white if any mismatch (shouldn't happen with our clamp)
    return Image.fromarray(collapsed)

def gray_image(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))

def leading_nonwhite(gray: np.ndarray, white_threshold=245) -> int:
    """
    First y where row is 'not mostly white' (has text/graphics).
    """
    for y in range(gray.shape[0]):
        if np.mean(gray[y]) < white_threshold:
            return y
    return gray.shape[0]  # all white

def trailing_nonwhite(gray: np.ndarray, white_threshold=245) -> int:
    """
    Last y where row is 'not mostly white'. If all white returns -1.
    """
    for y in range(gray.shape[0]-1, -1, -1):
        if np.mean(gray[y]) < white_threshold:
            return y
    return -1

def find_safe_white_cut(gray: np.ndarray, target_y: int, window=20, run=3, white_threshold=245):
    """
    Find a small run of white rows near target_y to avoid slicing text.
    Searches [target_y-window : target_y+window] for >=run consecutive white rows.
    Returns the first row index AFTER that white run (so you cut on whitespace).
    If none found, return None.
    """
    h = gray.shape[0]
    a = max(0, target_y - window)
    b = min(h, target_y + window)

    consec = 0
    for y in range(a, b):
        if np.mean(gray[y]) >= white_threshold:
            consec += 1
            if consec >= run:
                return y  # cut after this white run
        else:
            consec = 0
    return None

# ---------------------------
# Main raster reflow with cross-page spill
# ---------------------------
def raster_reflow(
    input_pdf: str,
    output_pdf: str,
    patterns: list,
    *,
    zoom: float = 2.0,
    include_following_space: bool = True,
    white_threshold: int = 245,
    debug: bool = False
):
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]

    src = fitz.open(input_pdf)

    # 1) Render every page to an image
    page_imgs = []
    for p in src:
        pix = p.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        page_imgs.append(pil_from_pixmap(pix))

    # 2) Process each page: remove header blocks (line-accurate) in image space
    processed_imgs = []
    for i, p in enumerate(src):
        img = page_imgs[i]
        gray = gray_image(img)
        H, W = gray.shape

        lines = extract_lines_dict(p)
        if debug:
            print(f"[page {i+1}] lines={len(lines)}")

        # indices of lines that match patterns
        matched = []
        for idx, ln in enumerate(lines):
            if any(r.search(normalize_text(ln["text"])) for r in regs):
                matched.append(idx)

        if not matched:
            processed_imgs.append(img)
            if debug:
                print(f"[page {i+1}] no header matches")
            continue

        groups = group_consecutive(matched)
        if debug:
            print(f"[page {i+1}] groups: {groups}")

        # Remove groups top→bottom. Keep running pixel shift so later bands land correctly.
        total_shift_px = 0
        for g0, g1 in groups:
            # Compute band in PDF points
            band_top_pt = lines[g0]["y_top"]
            band_bottom_pt = lines[g1]["y_bottom"]

            # Optionally extend to the top of the next non-matched line
            if include_following_space:
                nxt = g1 + 1
                if nxt < len(lines):
                    band_bottom_pt = max(band_bottom_pt, lines[nxt]["y_top"])

            # Convert to pixels and adjust by prior deletions
            y0_px = int(band_top_pt * zoom - total_shift_px)
            y1_px = int(band_bottom_pt * zoom - total_shift_px)

            # Clamp
            y0_px = max(0, min(y0_px, H))
            y1_px = max(0, min(y1_px, H))

            if debug:
                print(f"[page {i+1}] remove rows [{y0_px}:{y1_px}] (pts {band_top_pt:.2f}->{band_bottom_pt:.2f})")

            # Delete and collapse
            img = delete_rows_and_collapse(img, y0_px, y1_px)

            # Update gray + shift for the next group
            removed_h = max(0, y1_px - y0_px)
            total_shift_px += removed_h

            gray = gray_image(img)
            H, W = gray.shape

        processed_imgs.append(img)

    # 3) Cross-page spill: after collapsing page N, move the top block into previous page if there is space.
    final_imgs = []
    for i, img in enumerate(processed_imgs):
        gray = gray_image(img)
        H, W = gray.shape

        # Leading content on current page (top-most non-white row)
        top_nonwhite = leading_nonwhite(gray, white_threshold)

        # If this is the first page, no spill possible
        if i == 0 or top_nonwhite >= H:
            final_imgs.append(img)
            continue

        # Compute available space on previous page (bottom white margin)
        prev_img = final_imgs[-1]
        prev_gray = gray_image(prev_img)
        pH, pW = prev_gray.shape
        prev_bottom_nonwhite = trailing_nonwhite(prev_gray, white_threshold)
        available = max(0, pH - 1 - prev_bottom_nonwhite - 2)  # 2px padding

        if available <= 0 or top_nonwhite <= 0:
            final_imgs.append(img)
            continue

        # We want to move up to 'available' rows from the very top
        target_cut = top_nonwhite + min(available, H - top_nonwhite)

        # Find a safe white cut near target_cut to avoid slicing text
        safe_cut = find_safe_white_cut(gray, target_cut, window=30, run=3, white_threshold=white_threshold)

        if safe_cut is None:
            # fall back: if there's lots of white at the top, just move that white (no content)
            white_cap = min(top_nonwhite, available)
            if white_cap <= 0:
                final_imgs.append(img)
                continue
            # Move [0 : safe_cut] into previous page WITH TRIM

            # Amount to trim from the top of the spill block
            trim_amount = min(TOP_MARGIN_PX, safe_cut - 1)

            # First, crop the spill block (0 → safe_cut)
            spill_full = img.crop((0, 0, W, safe_cut))

            # Now trim the top of it by trim_amount
            spill = spill_full.crop((0, trim_amount, W, safe_cut))

            # Height of the trimmed spill
            spill_h = spill.size[1]

            # Compose previous page + spill
            new_prev = Image.new("RGB", (pW, pH + spill_h), "white")
            new_prev.paste(prev_img, (0, 0))
            new_prev.paste(spill, (0, pH))

            # Remove the entire full spill region from current page
            new_curr = img.crop((0, safe_cut, W, H))

            final_imgs[-1] = new_prev
            final_imgs.append(new_curr)

        else:
            # Move [0 : safe_cut] into previous page
            spill = img.crop((0, 0, W, safe_cut))
            new_prev = Image.new("RGB", (pW, pH + safe_cut), "white")
            new_prev.paste(prev_img, (0, 0))
            new_prev.paste(spill, (0, pH))
            # Remaining current page
            new_curr = img.crop((0, safe_cut, W, H))
            final_imgs[-1] = new_prev
            final_imgs.append(new_curr)

    # 4) Save all images as a PDF
    out = fitz.open()
    for img in final_imgs:
        w, h = img.size
        page = out.new_page(width=w, height=h)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        page.insert_image(page.rect, stream=bio.getvalue())
    out.save(output_pdf)
    out.close()

# ---------------------------
# Example usage
# ---------------------------
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

    raster_reflow(
        input_pdf="FakeStackedEmailThread.pdf",
        output_pdf="Cleaned_FakeStackedEmailThread3.pdf",
        patterns=patterns,
        zoom=1.0,
        include_following_space=True,
        white_threshold=245,
        debug=True,
    )


