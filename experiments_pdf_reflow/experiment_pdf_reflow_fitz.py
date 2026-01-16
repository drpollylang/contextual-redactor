# import fitz  # PyMuPDF
# import re
# import sys
# from pathlib import Path

# import unicodedata


# # This is required to normalize pdf text for matching (removes any formatting idiosyncrasies that 
# # would prevent regex matches)
# def normalize_pdf_text(t):
#     # Normalize unicode (fixes fancy dashes, quotes, ligatures, etc.)
#     t = unicodedata.normalize("NFKC", t)

#     # Convert all whitespace to single spaces
#     t = re.sub(r"\s+", " ", t)

#     # Remove soft hyphens or hyphen‑linebreak combinations
#     t = re.sub(r"-\s+", "", t)

#     # Trim
#     return t.strip()


# def remove_blocks_and_whitespace(input_pdf, output_pdf, remove_text_patterns=None):
#     """
#     Remove text blocks matching given patterns and clean extra whitespace.
    
#     :param input_pdf: Path to input PDF file
#     :param output_pdf: Path to save cleaned PDF
#     :param remove_text_patterns: List of regex patterns for text to remove
#     """
#     try:
#         doc = fitz.open(input_pdf)
#     except Exception as e:
#         print(f"Error opening PDF: {e}")
#         sys.exit(1)

#     if remove_text_patterns is None:
#         remove_text_patterns = []

#     for page_num, page in enumerate(doc, start=1):
#         blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no, block_type)
#         # for b in blocks:
#         #     x0, y0, x1, y1, text, *_ = b
#         #     # clean_text = re.sub(r"\s+", " ", text).strip()
#         #     normalized_text = normalize_pdf_text(text)

#         #     if any(re.search(pattern, normalized_text, re.IGNORECASE) for pattern in remove_text_patterns):
#         #         # redact only the unwanted text
#         #         page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1))
#         #         page.apply_redactions()
#         #         space_height = y1 - y0
#         #         # shift_page_content_preserve_layout(doc, page.number, y1, space_height)                              
#         #         # shift_after_gap_preserve_layout(
#         #         #         doc,
#         #         #         page_index=page.number,                    # zero-based
#         #         #         gap_top=y0,
#         #         #         gap_bottom=y1,
#         #         #         spill_to_previous_if_space=True  # optional
#         #         #     )
#         #         collapse_gap_by_compositing(
#         #             doc,
#         #             page_index=page.number,                    # zero-based
#         #             gap_top=y0,
#         #             gap_bottom=y1,
#         #             fill_bottom=True  # optional
#         #         )
#         #     else:
#         #         # DO NOT redact — just leave the block alone
#         #         pass
        
        
#         # === Phase 1: identify rects ===
#         rects_to_remove = []

#         for b in blocks:
#             x0, y0, x1, y1, text, *_ = b
#             normalized = normalize_pdf_text(text)

#             if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in remove_text_patterns):
#                 rects_to_remove.append((x0, y0, x1, y1))

#         # === Phase 2: apply all redactions ===
#         for rect in rects_to_remove:
#             page.add_redact_annot(fitz.Rect(*rect), fill=(1,1,1))

#         page.apply_redactions()

#         # IMPORTANT: get stable page index AFTER redactions
#         page_index = page.number

#         # === Phase 3: collapse gaps top-to-bottom ===
#         for (x0, y0, x1, y1) in sorted(rects_to_remove, key=lambda r: r[1]):
#             collapse_gap_by_compositing(
#                 doc,
#                 page_index=page_index, 
#                 gap_top=y0,
#                 gap_bottom=y1,
#                 fill_bottom=True
#             )



#     try:
#         doc.save(output_pdf)
#         print(f"✅ Cleaned PDF saved to: {output_pdf}")
#     except Exception as e:
#         print(f"Error saving PDF: {e}")
#     finally:
#         doc.close()




# # def shift_page_content_preserve_layout(doc, page_number, y_threshold, shift_amount):
# #     """
# #     Shift all text spans on a page upward, preserving original layout.
# #     Spill onto previous page ONLY if there is room.
# #     """

# #     page = doc[page_number]
# #     prev_page = doc[page_number - 1] if page_number > 0 else None

# #     # Extract raw spans
# #     raw = page.get_text("rawdict")
# #     blocks = raw["blocks"]

# #     # Extract previous page content bottom
# #     prev_bottom = 0
# #     if prev_page:
# #         prev_raw = prev_page.get_text("rawdict")
# #         for b in prev_raw.get("blocks", []):
# #             for line in b.get("lines", []):
# #                 for span in line.get("spans", []):
# #                     prev_bottom = max(prev_bottom, span["bbox"][3])

# #     # Prepare new shape for redrawing content
# #     shape_page = page.new_shape()
# #     shape_prev = prev_page.new_shape() if prev_page else None

# #     # Clear the page (remove its content stream)
# #     page.clean_contents()

# #     page_height = page.rect.height

# #     # for b in blocks:
# #     #     for line in b.get("lines", []):
# #     #         for span in line.get("spans", []):
                
# #     #             # Skip non-text spans (images, paths, etc.)
# #     #             if "text" not in span:
# #     #                 continue

# #     #             x0, y0, x1, y1 = span["bbox"]
# #     #             text = span["text"]
# #     #             font = span["font"]
# #     #             size = span["size"]
# #     #             color = span["color"]

# #     #             # only shift spans BELOW the threshold
# #     #             if y0 >= y_threshold:
# #     #                 new_y = y0 - shift_amount
# #     #             else:
# #     #                 new_y = y0
    
# #     for b in blocks:
# #         for line in b.get("lines", []):
# #             for span in line.get("spans", []):

# #                 # Only shift textual spans
# #                 if "text" not in span:
# #                     continue

# #                 x0, y0, x1, y1 = span["bbox"]

# #                 # SHIFT DECISION:
# #                 # Use span baseline, not block geometry
# #                 if y0 > threshold:   # threshold = max baseline of removed content
# #                     new_y = y0 - shift_amount
# #                 else:
# #                     new_y = y0


# #                 # If new_y goes above page top, try moving to previous page
# #                 if prev_page and new_y < 0:
# #                     span_height = y1 - y0
# #                     available = prev_page.rect.height - prev_bottom - 10

# #                     if span_height <= available:
# #                         # Place span at the next free spot on previous page
# #                         draw_y = prev_bottom + size
# #                         shape_prev.insert_text(
# #                             fitz.Point(x0, draw_y),
# #                             text,
# #                             fontname=font,
# #                             fontsize=size,
# #                             color=color,
# #                         )
# #                         prev_bottom += span_height
# #                         continue
# #                     else:
# #                         # Doesn't fit — clamp to page top
# #                         new_y = size + 2

# #                 # draw span on the current page
# #                 shape_page.insert_text(
# #                     fitz.Point(x0, new_y),
# #                     text,
# #                     fontname=font,
# #                     fontsize=size,
# #                     color=color,
# #                 )

# #     # Commit drawings
# #     shape_page.commit()
# #     if shape_prev:
# #         shape_prev.commit()

# #     print(f"Shifted content on page {page_number}.")



# import fitz  # PyMuPDF

# def shift_after_gap_preserve_layout(
#     doc,
#     page_index: int,
#     gap_top: float,
#     gap_bottom: float,
#     *,
#     spill_to_previous_if_space: bool = False,
#     top_margin: float = 2.0,
#     epsilon: float = 0.5,
# ):
#     """
#     Collapse a vertical gap on a page by shifting all spans whose *baseline* is
#     below `gap_bottom` upward by (gap_bottom - gap_top), preserving layout.
    
#     Coordinates follow PyMuPDF screen coords: y increases downward.
#     gap_top < gap_bottom.

#     Parameters
#     ----------
#     doc : fitz.Document
#     page_index : int
#         0-based page index to modify.
#     gap_top : float
#         Y where the removed region starts.
#     gap_bottom : float
#         Y where the removed region ends.
#     spill_to_previous_if_space : bool
#         If True, spans that would move above the top margin are placed on the
#         previous page *only if they fit*. Otherwise they are clamped at the top.
#     top_margin : float
#         Minimum Y for the baseline when clamping to the top of the current page.
#     epsilon : float
#         Tolerance to catch spans that sit exactly on the boundary due to float noise.
#     """

#     assert gap_bottom > gap_top, "gap_bottom must be > gap_top"
#     page = doc[page_index]
#     prev_page = doc[page_index - 1] if (spill_to_previous_if_space and page_index > 0) else None

#     # How much we need to move up
#     shift_amount = gap_bottom - gap_top

#     # --- Collect spans from current page (raw, layout-preserving) ---
#     raw = page.get_text("rawdict")
#     blocks = raw.get("blocks", [])

#     # Optionally compute the "occupied bottom" of the previous page
#     prev_bottom = 0.0
#     if prev_page is not None:
#         prev_raw = prev_page.get_text("rawdict")
#         for b in prev_raw.get("blocks", []):
#             for ln in b.get("lines", []):
#                 for sp in ln.get("spans", []):
#                     if "text" not in sp:
#                         continue
#                     # For a conservative "occupied" measure, use bbox bottom
#                     prev_bottom = max(prev_bottom, sp["bbox"][3])

#     # We'll redraw everything on a clean page
#     shape_here = page.new_shape()
#     shape_prev = prev_page.new_shape() if prev_page is not None else None

#     # Clear all original content streams (text/images/paths) on this page
#     page.clean_contents()

#     page_height = page.rect.height

#     moved_count = 0
#     kept_count = 0
#     skipped_in_gap = 0

#     for b in blocks:
#         for ln in b.get("lines", []):
#             for sp in ln.get("spans", []):
#                 if "text" not in sp:
#                     # Non-text spans (images/vector). We'll handle text first;
#                     # You can extend later to move images/paths similarly.
#                     continue

#                 text = sp["text"]
#                 if not text:
#                     continue

#                 # Use the *baseline* for vertical logic, not the bbox
#                 x_base, y_base = sp["origin"]  # baseline start (x, y)
#                 x0, y0, x1, y1 = sp["bbox"]    # bbox if needed
#                 font = sp.get("font", "Helvetica")
#                 size = sp.get("size", 12)
#                 color = sp.get("color", 0)

#                 # 1) If span baseline is inside the removed gap, skip it (it's removed)
#                 if (y_base >= (gap_top - epsilon)) and (y_base <= (gap_bottom + epsilon)):
#                     skipped_in_gap += 1
#                     continue

#                 # 2) If span baseline is below the gap bottom, shift it up by gap height
#                 if y_base > (gap_bottom + epsilon):
#                     new_y = y_base - shift_amount

#                     # If we would go above the top margin, consider previous-page spill
#                     if new_y < top_margin and prev_page is not None:
#                         # Check whether this span fits on the previous page
#                         span_height = y1 - y0  # approximate occupied height
#                         available = prev_page.rect.height - prev_bottom - top_margin
#                         if span_height <= available:
#                             # Place at the next free baseline position.
#                             # A simple (and workable) choice: baseline near bbox bottom.
#                             # We approximate baseline as prev_bottom - descender;
#                             # as we don't have detailed metrics, place it just above prev_bottom.
#                             place_baseline = max(prev_bottom + (span_height * 0.85), top_margin + size)
#                             shape_prev.insert_text(
#                                 fitz.Point(x_base, place_baseline),
#                                 text,
#                                 fontname=font,
#                                 fontsize=size,
#                                 color=color,
#                             )
#                             prev_bottom = max(prev_bottom, place_baseline + (span_height * 0.15))
#                             moved_count += 1
#                             continue
#                         else:
#                             # Not enough space: clamp on current page
#                             new_y = top_margin

#                     elif new_y < top_margin:
#                         new_y = top_margin

#                     # Draw shifted on the same page
#                     shape_here.insert_text(
#                         fitz.Point(x_base, new_y),
#                         text,
#                         fontname=font,
#                         fontsize=size,
#                         color=color,
#                     )
#                     moved_count += 1

#                 else:
#                     # 3) Above the gap: keep as-is
#                     shape_here.insert_text(
#                         fitz.Point(x_base, y_base),
#                         text,
#                         fontname=font,
#                         fontsize=size,
#                         color=color,
#                     )
#                     kept_count += 1

#     # Commit drawings
#     shape_here.commit()
#     if shape_prev is not None:
#         shape_prev.commit()

#     # Helpful diagnostics in your console
#     print(
#         f"[shift_after_gap] page={page_index+1} gap=({gap_top:.2f},{gap_bottom:.2f}) "
#         f"shift={shift_amount:.2f} moved={moved_count} kept={kept_count} removed_in_gap={skipped_in_gap}"
#     )




# def collapse_gap_by_compositing(doc, page_index, gap_top, gap_bottom, *, fill_bottom=True):
#     """
#     Collapse a vertical gap on a page by shifting everything below upward.
#     Works on all PDFs (text, images, vectors, forms) because it composites
#     rendered content rather than rewriting text spans.
#     """

#     assert gap_bottom > gap_top, "gap_bottom must be > gap_top"

#     src = doc[page_index]
#     w, h = src.rect.width, src.rect.height
#     shift = gap_bottom - gap_top

#     # --- 1. Create a separate temporary document with a copy of the page ---
#     temp_doc = fitz.open()                # new empty PDF
#     temp_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
#     src_clone = temp_doc[0]               # the cloned page in its own doc

#     # --- 2. Create a new blank page to draw the composited result ---
#     composed = doc.new_page(page_index + 1, width=w, height=h)

#     # --- 3. Copy the upper region unchanged ---
#     upper_clip = fitz.Rect(0, 0, w, gap_top)
#     composed.show_pdf_page(upper_clip, temp_doc, 0, clip=upper_clip)

#     # --- 4. Copy the lower region shifted upward ---
#     lower_clip = fitz.Rect(0, gap_bottom, w, h)
#     lower_target = fitz.Rect(0, gap_bottom - shift, w, h - shift)

#     composed.show_pdf_page(lower_target, temp_doc, 0, clip=lower_clip)

#     # --- 5. Optionally fill the bottom gap with white ---
#     if fill_bottom and shift > 0:
#         shape = composed.new_shape()
#         shape.draw_rect(
#             fitz.Rect(0, h - shift, w, h),
#             # fill=(1, 1, 1),
#         )
#         shape.commit()

#     # --- 6. Delete the original page ---
#     doc.delete_page(page_index)

#     # temp_doc automatically GC’d
#     print(f"[collapse_gap] page={page_index+1} shift={shift}")



# # ----------- v2 -------------
# # def remove_lines_and_collapse_whitespace(input_pdf, output_pdf, remove_pattern):
# #     doc = fitz.open(input_pdf)

# #     for page in doc:
# #         blocks = page.get_text("blocks")

# #         # Keep track of vertical offset caused by removed blocks
# #         y_offset = 0

# #         for b in blocks:
# #             x0, y0, x1, y1, text, *_ = b
# #             clean = re.sub(r"\s+", " ", text).strip()

# #             block_height = y1 - y0

# #             if re.match(remove_pattern, clean, flags=re.IGNORECASE):
# #                 # Remove this block — it will create vertical space
# #                 page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1))
# #                 y_offset += block_height
# #             else:
# #                 # Reinsert text shifted up by accumulated offset
# #                 if y_offset > 0:
# #                     new_rect = fitz.Rect(x0, y0 - y_offset, x1, y1 - y_offset)
# #                     page.add_redact_annot(fitz.Rect(x0, y0, x1, y1), fill=(1, 1, 1))
# #                     page.insert_textbox(new_rect, clean, fontsize=10)

# #         page.apply_redactions()

# #     doc.save(output_pdf)
# #     doc.close()
# #     print("Done.")



# if __name__ == "__main__":
#     input_file = "FakeStackedEmailThread.pdf"
#     output_file = "Cleaned_FakeStackedEmailThread.pdf"
    
#     # Patterns for blocks to remove (case-insensitive)
#     # patterns_to_remove = [
#     #     r"confidential",   # remove blocks containing "confidential"
#     #     r"page \d+ of \d+" # remove page numbering
#     # ]
#     # pattern = r"^From:\s*(.+)"
#     # pattern = r"^Just a quick update\s*—\s*the client has approved the revised design\. Please confirm if we\s+can move the development phase forward to next Monday$"
#     # pattern = r"^Hi Sarah,\nThat works for me. I\'ll have the initial setup ready by Friday.\nBest, James$"
#     # patterns_to_remove = [pattern]
#     # patterns_to_remove = [r"^From:\s*(.+)", r"^To:\s*(.+)", r"^Subject:\s*(.+)", r"^Date:\s*(.+)"]
#     # patterns_to_remove = [r"^Just\s*(.+)a\s*(.+)quick\s*(.+)update\s*(.+)", r"^From:\s*Sarah\s*(.+)", r"^To:\s*(.+)Team\s*(.+)", r"^Subject:\s*(.+)Project\s*(.+)Timeline\s*(.+)Update\s*(.+)", r"^Date:\s*(.+)10\s*(.+)Jan\s*(.+)2026\s*(.+)09\s*(.+)15\s*(.+)"]

#     pattern = r"From: Sarah \(Project Manager\) \n To: Team \n Subject: Project Timeline Update \n Date: 10 Jan 2026, 09:15 \n Hi Team, \n Just a quick update — the client has approved the revised design. Please confirm if we can move the development phase forward to next Monday. \n Thanks, \n Sarah".replace(" ", r"\s*(.+)")
#     patterns_to_remove = [pattern]

#     if not Path(input_file).exists():
#         print(f"❌ File not found: {input_file}")
#         sys.exit(1)

#     remove_blocks_and_whitespace(input_file, output_file, patterns_to_remove)
#     # remove_lines_and_collapse_whitespace(input_file, output_file, pattern)


# ------------------------------------------------------------------------------------------


import fitz
import re
import unicodedata


# ------------------------------------------------------------
# NORMALIZATION FOR PDF TEXT MATCHING
# ------------------------------------------------------------
def normalize_pdf_text(t: str) -> str:
    # normalize unicode dashes, ligatures, accents
    t = unicodedata.normalize("NFKC", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    # strip
    return t.strip()


# ------------------------------------------------------------
# FIND THE ACTUAL GAP AFTER REDACTION
# ------------------------------------------------------------
def find_actual_gap(page, removed_y1):
    """
    Given the bottom of a removed block (removed_y1), identify the FIRST
    block whose y0 is > removed_y1. The gap that should be collapsed is:
    gap = (removed_y1, next_block.y0)
    """
    blocks = page.get_text("blocks")
    candidates = [b for b in blocks if b[1] > removed_y1]

    if not candidates:
        return None

    next_top = min(b[1] for b in candidates)
    return (removed_y1, next_top)


# ------------------------------------------------------------
# COLLAPSE GAP VIA COMPOSITING (LAYOUT PRESERVING)
# ------------------------------------------------------------
def collapse_gap_by_compositing(doc, page_index, gap_top, gap_bottom, *, fill_bottom=True):
    """
    This version uses a temporary document clone to avoid the
    "source document must not equal target" restriction.

    Everything below (gap_bottom, page.height) is shifted upward by:
        shift = gap_bottom - gap_top
    """
    page = doc[page_index]
    w, h = page.rect.width, page.rect.height
    shift = gap_bottom - gap_top

    # 1: temp doc containing only this page
    temp = fitz.open()
    temp.insert_pdf(doc, from_page=page_index, to_page=page_index)

    # 2: compose into a fresh page immediately after original
    composed = doc.new_page(page_index + 1, width=w, height=h)

    # --- upper unchanged part ---
    upper_clip = fitz.Rect(0, 0, w, gap_top)
    composed.show_pdf_page(upper_clip, temp, 0, clip=upper_clip)

    # --- lower shifted part ---
    lower_clip = fitz.Rect(0, gap_bottom, w, h)
    lower_target = fitz.Rect(0, gap_bottom - shift, w, h - shift)
    composed.show_pdf_page(lower_target, temp, 0, clip=lower_clip)

    # --- fill bottom gap with white ---
    if fill_bottom and shift > 0:
        shape = composed.new_shape()
        shape.draw_rect(
            fitz.Rect(0, h - shift, w, h),
            # color=None,
            # fill=(1, 1, 1)
        )
        shape.commit()

    # 3: delete original page
    doc.delete_page(page_index)

    # Now the composed page occupies index = page_index


# ------------------------------------------------------------
# MAIN CLEANER
# ------------------------------------------------------------
def clean_pdf_remove_lines_and_collapse(input_pdf, output_pdf, patterns):
    doc = fitz.open(input_pdf)

    for page_index in range(len(doc)):
        page = doc[page_index]

        # --- PHASE 1: identify blocks to remove ---
        blocks = page.get_text("blocks")
        rects_to_remove = []

        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            norm = normalize_pdf_text(text)
            if any(re.search(p, norm, re.IGNORECASE) for p in patterns):
                rects_to_remove.append((x0, y0, x1, y1))

        if not rects_to_remove:
            continue  # no redactions on this page

        # --- PHASE 2: apply all redactions at once ---
        for rect in rects_to_remove:
            page.add_redact_annot(fitz.Rect(*rect), fill=(1, 1, 1))

        page.apply_redactions()   # IMPORTANT

        # Refresh the page object after redactions
        page = doc[page_index]

        # --- PHASE 3: collapse gaps top-to-bottom ---
        rects_sorted = sorted(rects_to_remove, key=lambda r: r[1])

        for (x0, y0, x1, y1) in rects_sorted:
            # Compute actual gap AFTER redaction
            gap = find_actual_gap(page, y1)
            if not gap:
                continue

            gap_top, gap_bottom = gap
            if gap_bottom - gap_top <= 0:
                continue

            collapse_gap_by_compositing(
                doc, page_index, gap_top, gap_bottom, fill_bottom=True
            )

            # Page was replaced; reacquire it
            page = doc[page_index]

    doc.save(output_pdf)
    doc.close()


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
# if __name__ == "__main__":
#     # patterns = [r"^From:\s*"]
#     patterns = [r"^From:\s*(.+)", r"^To:\s*(.+)", r"^Subject:\s*(.+)", r"^Date:\s*(.+)"]

#     clean_pdf_remove_lines_and_collapse(
#         "FakeStackedEmailThread.pdf",
#         "Cleaned_FakeStackedEmailThread.pdf",
#         patterns
#     )




doc = fitz.open("FakeStackedEmailThread.pdf")
page = doc[0]

print("=== TEXT RAWDICT ===")
print(page.get_text("rawdict"))

print("\n=== XOBJECTS ===")
for x in page.get_xobjects():
    print(x)

print("\n=== DRAWINGS ===")
for d in page.get_drawings():
    print(d)

print("\n=== IMAGES ===")
print(page.get_images(full=True))
