import streamlit as st
import fitz  # PyMuPDF
import os
import time
import requests, base64
from io import BytesIO
from collections import defaultdict
from PIL import ImageDraw, Image
from streamlit_drawable_canvas import st_canvas
from redaction_logic import analyse_document_for_redactions, analyse_document_structure, analyse_document_for_duplicate_emails
from pdf_processor import PDFProcessor
from utils import get_original_pdf_images
from pdf_edit_client import call_pdf_edit_api, rect_to_payload, canvas_obj_to_page_rect

st.set_page_config(page_title="AI Document Redactor", layout="wide")

PREVIEW_DPI = 150
# Define a fixed display width for the canvas to prevent overflow
CANVAS_DISPLAY_WIDTH = 800

API_URL = st.secrets.get("PDF_API_URL", "http://localhost:8008")  # set via .streamlit/secrets.toml


# def fetch_overlay_image(pdf_bytes: bytes, page_index: int, rect: dict, api_url: str):
#     payload = {
#         "pdf_b64": base64.b64encode(pdf_bytes).decode("utf-8"),
#         "page_index": page_index,
#         "rect": rect,
#         "dpi": 120
#     }
#     r = requests.post(f"{api_url.rstrip('/')}/v1/debug/overlay", json=payload, timeout=60)
#     r.raise_for_status()
#     return Image.open(BytesIO(r.content))


def main():
    """The main function that runs the Streamlit application."""

    # Initialise all session state variables
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'final_pdf_path' not in st.session_state:
        st.session_state.final_pdf_path = None
    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = None
    if 'approval_state' not in st.session_state:
        st.session_state.approval_state = {}
    if 'original_pdf_images' not in st.session_state:
        st.session_state.original_pdf_images = []
    if 'user_context' not in st.session_state:
        st.session_state.user_context = ""
    if 'manual_rects' not in st.session_state:
        st.session_state.manual_rects = defaultdict(list)
    if 'drawing_mode' not in st.session_state:
        st.session_state.drawing_mode = "rect"
    if 'active_page_index' not in st.session_state:
        st.session_state.active_page_index = 0
    if 'last_promoted_ids' not in st.session_state:
        st.session_state.last_promoted_ids = []
    if 'time_elapsed' not in st.session_state:
        st.session_state.time_elapsed = None
    if 'duplicate_emails' not in st.session_state:
        st.session_state.duplicate_emails = []
    if 'duplicate_email_state' not in st.session_state:
        st.session_state.duplicate_email_state = {}
    if "removed_sections" not in st.session_state:
        st.session_state.removed_sections = {}  # {page_index: [fitz.Rect, ...]}


    # --- Main App UI ---
    st.title("AI-Powered Document Redaction Tool")
    st.write("Upload a PDF to redact. Use the options below to guide the AI and refine the results.")

    temp_dir = "temp_docs"
    output_dir = "redacted_docs"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

    st.text_area(
        "Provide specific redaction instructions for this document (optional):",
        placeholder=(
            "The AI redacts common personal data by default. Use this box to provide exceptions or new rules.\n\n"
        ),
        height=100,
        key='user_context'
    )

    if uploaded_file is not None:
        st.session_state.pdf_bytes = uploaded_file.getvalue()
        input_pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Analyse Document"):
            is_new_file = st.session_state.processed_file != input_pdf_path
            if is_new_file or not st.session_state.original_pdf_images:
                with st.spinner("Rendering document preview..."):
                    st.session_state.original_pdf_images = get_original_pdf_images(input_pdf_path)
            st.session_state.suggestions = []
            st.session_state.duplicate_emails = []
            st.session_state.approval_state = {}
            st.session_state.duplicate_email_state = {}
            st.session_state.final_pdf_path = None
            st.session_state.manual_rects = defaultdict(list)
            st.session_state.active_page_index = 0
            # Start timer
            start = time.perf_counter()
            
            with st.spinner("Analysing document with your instructions..."):
                analysed_document = analyse_document_structure(input_pdf_path)
                suggestions = analyse_document_for_redactions(analysed_document, st.session_state.user_context)
                duplicate_emails = analyse_document_for_duplicate_emails(analysed_document)
                st.session_state.suggestions = suggestions
                st.session_state.duplicate_emails = duplicate_emails
                st.session_state.processed_file = input_pdf_path
                st.session_state.approval_state = {s['id']: True for s in suggestions}
                st.session_state.duplicate_email_state = {d['id']: True for d in duplicate_emails}
                st.session_state.original_pdf_images = get_original_pdf_images(input_pdf_path)
            
            # Stop timer
            st.session_state.time_elapsed = time.perf_counter() - start
            
            if suggestions:
                st.success(f"Analysis complete! Found {len(suggestions)} total instances to review.")
            else:
                st.warning("Analysis complete, but no sensitive information was found.")
            
    # Print time taken to analyse instructions
    if st.session_state.time_elapsed is not None:
        st.success(f"Completed in {st.session_state.time_elapsed:.3f} seconds")
        # st.metric(label="Time Elapsed", value=f"{st.session_state.time_elapsed:.3f}s")

    
    def apply_removals_for_page(page_index: int, rects: list, api_url: str):
        """
        rects: list[fitz.Rect] in page coordinates to remove on this page
        """
        if not rects:
            st.warning("No sections to remove on this page.")
            return

        # Ensure bytes in state (defensive)
        if "pdf_bytes" not in st.session_state or not isinstance(st.session_state.pdf_bytes, (bytes, bytearray)):
            st.error("PDF not loaded correctly in memory. Please re-upload.")
            return

        operations = [{
            "page_index": page_index,
            "rects": [rect_to_payload(r) for r in sorted(rects, key=lambda r: r.y0)]
        }]

        with st.spinner("Applying removals and reflow..."):
            try:
                new_pdf_bytes = call_pdf_edit_api(st.session_state.pdf_bytes, operations, api_url=API_URL, font_mode="auto")
            except Exception as e:
                st.error(f"API error: {e}")
                return

        # Replace working bytes in session
        st.session_state.pdf_bytes = new_pdf_bytes

        # Overwrite the temp file so your preview renders the updated document
        if "processed_file" in st.session_state and st.session_state.processed_file:
            with open(st.session_state.processed_file, "wb") as f:
                f.write(new_pdf_bytes)

        # Re-render page images from the updated file
        with st.spinner("Refreshing preview..."):
            st.session_state.original_pdf_images = get_original_pdf_images(st.session_state.processed_file)

        st.success("Removals applied.")
        st.rerun()

    
    def ensure_pdf_bytes_in_state():
        """Make sure st.session_state.pdf_bytes exists and is bytes."""
        if "pdf_bytes" in st.session_state and isinstance(st.session_state.pdf_bytes, (bytes, bytearray)):
            return
        # Fallback: load from processed_file path if present
        if "processed_file" in st.session_state and st.session_state.processed_file:
            with open(st.session_state.processed_file, "rb") as f:
                st.session_state.pdf_bytes = f.read()
        else:
            st.error("No PDF loaded. Please upload a PDF.")
            st.stop()


    if st.session_state.suggestions or st.session_state.duplicate_emails:
        st.header("Review and Refine Redactions")
        col1, col2 = st.columns([1, 3])

        with col1:
            
            
            st.subheader("AI Suggestions")
            st.write("Control all instances of a term or expand to manage each one individually.")
            doc = fitz.open(st.session_state.processed_file)

            grouped_suggestions = defaultdict(list)
            for s in st.session_state.suggestions:
                grouped_suggestions[s['text']].append(s)

            for text, instances in grouped_suggestions.items():
                all_ids = [inst['id'] for inst in instances]
                all_checked = all(st.session_state.approval_state.get(id, False) for id in all_ids)
                master_state = st.checkbox(
                    f"**{text}** ({instances[0]['category']}) - {len(instances)} instance(s)",
                    value=all_checked, key=f"master_{text}"
                )
                if master_state != all_checked:
                    for id in all_ids:
                        st.session_state.approval_state[id] = master_state
                    st.rerun()

                with st.expander("Show individual occurrences"):
                    for instance in instances:
                        if instance['category'] != 'Manual Selection':
                            context = instance['context']
                            start = max(0, context.find(text) - 30)
                            end = min(len(context), start + len(text) + 60)
                            label = f"Pg {instance['page_num'] + 1}: ...{context[start:end]}..." 
                        else:
                            page = doc[instance['page_num']]
                            if instance['rects']:
                                main_rect = instance['rects'][0]
                                context_rect = main_rect + (-80, -5, 80, 5)
                                words_in_area = page.get_text("words", clip=context_rect)
                                words_in_area.sort(key=lambda w: w[0])
                                context_snippet = " ".join(w[4] for w in words_in_area)
                                highlighted_snippet = context_snippet.replace(text, f"**{text}**")
                                label = f"Pg {instance['page_num'] + 1}: ...{highlighted_snippet}..."
                            else:
                                label = f"Pg {instance['page_num'] + 1}: (No context preview available)"
                        st.session_state.approval_state[instance['id']] = st.checkbox(
                            label, value=st.session_state.approval_state.get(instance['id'], True), key=f"cb_{instance['id']}"
                        )
            doc.close()

            # if st.session_state.duplicate_emails:
            #     st.subheader("Duplicate Emails Detected")
            #     st.write("Review duplicate emails. You can remove their content and refactor the document.")
            #     doc = fitz.open(st.session_state.processed_file)
            #     remove_state = {}
            #     for dup in st.session_state.duplicate_emails:
            #         context = dup['context']
            #         text = dup['text']
            #         start = max(0, context.find(text) - 30)
            #         end = min(len(context), start + len(text) + 60)
            #         label = f"Pg {dup['page_num'] + 1}: ...{context[start:end]}..."
            #         remove_state[dup["id"]] = st.checkbox(
            #             f"Remove duplicate: {label}", value=st.session_state.duplicate_email_state.get(dup['id'], True), key=f"dup_{dup['id']}"
            #         )
            #         # remove_state[dup["id"]] = st.checkbox(f"Remove duplicate {dup['id']}", value=True, key=f"dupemail_{page_index}_{dup['id']}")
            #         st.session_state.duplicate_email_state[dup['id']] = remove_state
                
            #     # ------ add a button to remove duplicate emails and refactor the document here-----
            #     # Build operations payload from UNCHECKED sections
            #     checked_sections = [d for d in st.session_state.duplicate_emails if remove_state[d["id"]]]
            #     rects_payload = [rect_to_payload(d["rects"][0]) for d in sorted(checked_sections, key=lambda x: x["rects"][0].y0)]

            #     if st.button("Apply removals (checked sections) on this page", key=f"apply_{st.session_state.active_page_index}"):
            #         if not rects_payload:
            #             st.warning("No sections to remove.")
            #             return

            #         operations = [{"page_index": st.session_state.active_page_index, "rects": rects_payload}]
            #         try:
            #             new_pdf = call_pdf_edit_api(st.session_state.processed_file, operations, API_URL)
            #         except Exception as e:
            #             st.error(f"API error: {e}")
            #             return

            #         # Replace working PDF in state and rerun
            #         st.session_state.processed_file = new_pdf
            #         st.rerun()

            #     doc.close()

            
            if st.session_state.duplicate_emails:
                st.subheader("Duplicate Emails Detected")
                st.write("Review duplicate emails. You can remove their content and refactor the document.")

                # Ensure we have in-memory bytes ready for the API
                ensure_pdf_bytes_in_state()

                # (Optional) open the doc if you need page sizes; not required for this block
                # doc = fitz.open("pdf", st.session_state.pdf_bytes)

                # Build checkbox UI and remember per-duplicate state
                remove_state = {}
                for dup in st.session_state.duplicate_emails:
                    context = dup.get('context', '') or ''
                    text = dup.get('text', '') or ''
                    start = max(0, context.find(text) - 30) if text else 0
                    end = min(len(context), start + len(text) + 60) if text else min(len(context), 90)

                    # dup['page_num'] should be 0-based; adjust label only
                    # label = f"Pg {dup['page_num'] + 1}: ...{context[start:end]}..."
                    pages = sorted({r["page_num"] for r in dup["rects"]})
                    page_list = ", ".join(str(p+1) for p in pages)
                    label = f"Pg(s) {page_list}: ...{context[start:end]}..."


                    # Default value: remove = True (checked means remove)
                    default_val = st.session_state.duplicate_email_state.get(dup['id'], True)
                    remove_state[dup["id"]] = st.checkbox(
                        f"Remove duplicate: {label}",
                        value=default_val,
                        key=f"dup_{dup['id']}"
                    )
                    # Store just the boolean for this id (not the whole dict)
                    st.session_state.duplicate_email_state[dup['id']] = remove_state[dup['id']]

                # ---- Build operations payload from CHECKED sections (to remove) ----
                # Filter the duplicates the user checked for removal and that have rects
                to_remove = [
                    d for d in st.session_state.duplicate_emails
                    if remove_state.get(d["id"], False) and d.get("rects") and len(d["rects"]) > 0
                ]

                # Group rects by page, and sort rects top->bottom within page
                rects_by_page = defaultdict(list)
                # for d in to_remove:
                #     page_idx = d["page_num"]  # assume 0-based page index
                #     r = d["rects"][0]         # fitz.Rect in PAGE coordinates
                #     rects_by_page[page_idx].append(r)
                to_remove = [
                    d for d in st.session_state.duplicate_emails
                    if st.session_state.duplicate_email_state.get(d["id"], False)
                ]

                rects_by_page = defaultdict(list)
                for d in to_remove:
                    for entry in d["rects"]:
                        page_idx = entry["page_num"]
                        rects_by_page[page_idx].append(entry["rect"])


                operations = []
                for page_idx, rects in rects_by_page.items():
                    rects_sorted = sorted(rects, key=lambda r: r.y0)
                    operations.append({
                        "page_index": page_idx,
                        "rects": [rect_to_payload(r) for r in rects_sorted]
                    })
                  
                # 🔍 DEBUG HERE: PRINT EXACT RECTS YOU ARE SENDING
                st.write("DEBUG: Rects going to API (before clicking):")
                for rp in rects_by_page.items():
                    st.write(rp)
                st.write("FINAL OPERATIONS:", operations)

                # Button applies removals across all pages represented in 'operations'
                if st.button("Apply removals (checked sections)", key=f"apply_dupes"):
                    if not operations:
                        st.warning("No sections to remove.")
                        st.stop()

                    # Call API with BYTES, not a path
                    try:
                        new_pdf_bytes = call_pdf_edit_api(
                            st.session_state.pdf_bytes,  # <-- BYTES here
                            operations,
                            API_URL
                        )
                    except Exception as e:
                        st.error(f"API error: {e}")
                        st.stop()

                    # Update canonical in-memory bytes
                    st.session_state.pdf_bytes = new_pdf_bytes

                    # Mirror to disk (so any file-path-based preview continues to work)
                    if "processed_file" in st.session_state and st.session_state.processed_file:
                        with open(st.session_state.processed_file, "wb") as f:
                            f.write(new_pdf_bytes)

                    # Re-render previews after modification
                    with st.spinner("Refreshing preview..."):
                        st.session_state.original_pdf_images = get_original_pdf_images(st.session_state.processed_file)

                    st.success("Removals applied and document reflowed.")
                    st.rerun()

                # if you opened doc above, close it
                # doc.close()

                # if st.checkbox("Show debug overlay for first rect on this page"):
                #     if operations and operations[0]["rects"]:
                #         overlay = fetch_overlay_image(st.session_state.pdf_bytes, operations[0]["page_index"], operations[0]["rects"][0], API_URL)
                #         st.image(overlay, caption="Debug overlay: red=remove rect, green=lines to remove", use_column_width=True)



        with col2:
            st.subheader("Interactive Document Preview")
            if st.session_state.original_pdf_images:

                # Navigation Logic start
                total_pages = len(st.session_state.original_pdf_images)
                nav_cols = st.columns([1, 1, 6, 1, 1]) # Create columns for layout

                # Previous Page Button
                with nav_cols[0]:
                    if st.button("⬅️", use_container_width=True, disabled=(st.session_state.active_page_index == 0)):
                        st.session_state.active_page_index -= 1
                        st.rerun()

                with nav_cols[1]:
                    if st.button("Prev", use_container_width=True, disabled=(st.session_state.active_page_index == 0)):
                        st.session_state.active_page_index -= 1
                        st.rerun()
                
                # Page Counter Display
                with nav_cols[2]:
                    st.markdown(f"<p style='text-align: center; font-weight: bold;'>Page {st.session_state.active_page_index + 1} of {total_pages}</p>", unsafe_allow_html=True)
                
                # Next Page Button
                with nav_cols[3]:
                    if st.button("Next", use_container_width=True, disabled=(st.session_state.active_page_index >= total_pages - 1)):
                        st.session_state.active_page_index += 1
                        st.rerun()

                with nav_cols[4]:
                    if st.button("➡️", use_container_width=True, disabled=(st.session_state.active_page_index >= total_pages - 1)):
                        st.session_state.active_page_index += 1
                        st.rerun()

                page_index = st.session_state.active_page_index
                # --- END OF NAVIGATION LOGIC ---

                tool_cols = st.columns(3)
                with tool_cols[0]:
                    # Toggle between draw and edit mode
                    mode_toggle = st.checkbox("Enable Edit/Delete Mode")#, value=(st.session_state.drawing_mode == "transform"))
                    st.session_state.drawing_mode = "transform" if mode_toggle else "rect"

                with tool_cols[1]:
                    can_promote = len(st.session_state.manual_rects.get(page_index, [])) > 0
                    if st.button("Redact all occurrences of last drawn shape", disabled=not can_promote, use_container_width=True):

                        st.session_state.last_promoted_ids = []
                        
                        last_drawn_shape = st.session_state.manual_rects[page_index][-1]#.pop()
                        
                        doc = fitz.open(st.session_state.processed_file)
                        page = doc[page_index]
                        
                        # Scale canvas coords to PDF coords
                        original_img_width = st.session_state.original_pdf_images[page_index].width
                        scaling_factor = (72.0 / PREVIEW_DPI) * (original_img_width / CANVAS_DISPLAY_WIDTH)
                        
                        x1, y1 = last_drawn_shape["left"], last_drawn_shape["top"]
                        x2, y2 = x1 + last_drawn_shape["width"], y1 + last_drawn_shape["height"]
                        pdf_rect = fitz.Rect(x1 * scaling_factor, y1 * scaling_factor, x2 * scaling_factor, y2 * scaling_factor)
                        
                        # Extract text under the drawn rectangle
                        text_to_find = page.get_text("text", clip=pdf_rect).strip()
                        
                        if text_to_find:
                            newly_promoted_ids = []
                            # Search all pages for this text
                            for p_num, p_obj in enumerate(doc):
                                words_on_page = p_obj.get_text("words")  
                                
                                for word_index, word_tuple in enumerate(words_on_page):
                                    x0, y0, x1, y1, word_text = word_tuple[:5]

                                    normalised_word = word_text.lower().rstrip(".,;:!?'’s()")
                                    normalised_text_to_find = text_to_find.lower()

                                    if normalised_word == normalised_text_to_find:
                                        # It's a valid whole-word match!
                                        area_rect = fitz.Rect(x0, y0, x1, y1)
                                        new_id = f"promo_{text_to_find.replace(' ','_')}_{p_num}_{word_index}"
                                    
                                        new_suggestion = {
                                            'id': new_id,
                                            'text': text_to_find,
                                            'category': 'Manual Selection',
                                            'reasoning': 'Identified by user and found in all occurrences.',
                                            'context': p_obj.get_text("text"), # The whole page as context
                                            'page_num': p_num,
                                            'rects': [area_rect] # PyMuPDF returns a list of fitz.Rect
                                        }

                                        # Avoid adding duplicates
                                        if not any(s['id'] == new_id for s in st.session_state.suggestions):
                                            st.session_state.suggestions.append(new_suggestion)
                                            st.session_state.approval_state[new_id] = True
                                            newly_promoted_ids.append(new_id)
                            st.session_state.last_promoted_ids = newly_promoted_ids
                        doc.close()
                        st.rerun()
                
                with tool_cols[2]:
                    # Undo button is only active if there's a promotion to undo
                    can_undo = len(st.session_state.last_promoted_ids) > 0
                    if st.button("Undo last 'Redact All'", disabled=not can_undo, use_container_width=True):
                        ids_to_remove = st.session_state.last_promoted_ids
                        st.session_state.suggestions = [s for s in st.session_state.suggestions if s['id'] not in ids_to_remove]
                        for id_to_remove in ids_to_remove:
                            st.session_state.approval_state.pop(id_to_remove, None)
                        st.session_state.last_promoted_ids = [] # Clear the undo buffer
                        st.rerun()

                # DYNAMIC BACKGROUND & SCALING LOGIC
                original_image = st.session_state.original_pdf_images[page_index]
                
                # Calculate scaling factor for display
                display_scaling_factor = CANVAS_DISPLAY_WIDTH / original_image.width
                display_height = int(original_image.height * display_scaling_factor)
                
                # Create the resized image that will be displayed
                display_image = original_image.resize((CANVAS_DISPLAY_WIDTH, display_height))
                
                # Create a transparent layer for drawing redactions
                transparent_layer = Image.new("RGBA", display_image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(transparent_layer)

                # Define redaction colour and opacity
                redaction_fill = (255, 255, 0, 102)  # Semi-transparent yellow

                # Draw approved AI suggestions, scaled to the display size
                approved_ai_suggestions = [
                    s for s in st.session_state.suggestions 
                    if st.session_state.approval_state.get(s['id']) and s['page_num'] == page_index
                ]
                dpi_to_display_scaling = (PREVIEW_DPI / 72.0) * display_scaling_factor
                for suggestion in approved_ai_suggestions:
                    for rect in suggestion.get('rects', []):
                        # print(f"Suggestion rect: {rect}")
                        scaled_rect = (
                            rect.x0 * dpi_to_display_scaling, rect.y0 * dpi_to_display_scaling,
                            rect.x1 * dpi_to_display_scaling, rect.y1 * dpi_to_display_scaling
                        )
                        draw.rectangle(scaled_rect, fill=redaction_fill)

                # Draw duplicate email boxes in a different color (e.g., semi-transparent red)
                duplicate_email_fill = (255, 0, 0, 102)  # Semi-transparent red
                # for d in st.session_state.duplicate_emails:
                #         print(f"Page index: {page_index}, Duplicate email id and page num: {d['id']} / {d['page_num']}")
                # approved_duplicate_emails = [
                #     d for d in st.session_state.duplicate_emails
                #     if st.session_state.duplicate_email_state.get(d['id']) and d['page_num'] == page_index
                # ]
                approved_duplicate_emails = [
                    d for d in st.session_state.duplicate_emails
                    if st.session_state.duplicate_email_state.get(d['id'], False)
                    and any(r["page_num"] == page_index for r in d["rects"])
                ]
                # for dup in approved_duplicate_emails:
                #     print(f'dup: {dup}')
                #     for rect in dup.get('rects', []):
                #         # print(f"Duplicate email rect: {rect}")
                #         scaled_rect = (
                #             rect.x0 * dpi_to_display_scaling, rect.y0 * dpi_to_display_scaling,
                #             rect.x1 * dpi_to_display_scaling, rect.y1 * dpi_to_display_scaling
                #         )
                #         draw.rectangle(scaled_rect, fill=duplicate_email_fill)
                for dup in st.session_state.duplicate_emails:
                    if not st.session_state.duplicate_email_state.get(dup['id'], False):
                        continue

                    for item in dup["rects"]:
                        if item["page_num"] != page_index:
                            continue

                        rect = item["rect"]
                        scaled_rect = (
                            rect.x0 * dpi_to_display_scaling,
                            rect.y0 * dpi_to_display_scaling,
                            rect.x1 * dpi_to_display_scaling,
                            rect.y1 * dpi_to_display_scaling,
                        )
                        draw.rectangle(scaled_rect, fill=duplicate_email_fill)

                # Composite the AI suggestions and duplicate emails onto the background
                final_bg_image = Image.alpha_composite(display_image.convert("RGBA"), transparent_layer)
                
                st.info("In 'Draw' mode, create new redactions. In 'Edit/Delete' mode, you can move, resize, or double-click to delete shapes.")
                
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 0, 0.4)",
                    stroke_width=0,
                    background_image=final_bg_image,
                    update_streamlit=True,
                    height=display_height,
                    width=CANVAS_DISPLAY_WIDTH,
                    drawing_mode=st.session_state.drawing_mode,
                    # Load initial drawing from state to persist edits
                    initial_drawing={"objects": st.session_state.manual_rects.get(page_index, [])},
                    key=f"canvas_{page_index}",
                )

                if canvas_result.json_data is not None:
                    # Only update state if the drawn objects have actually changed
                    if canvas_result.json_data.get("objects") != st.session_state.manual_rects.get(page_index, []):
                        st.session_state.manual_rects[page_index] = canvas_result.json_data.get("objects", [])
                        st.rerun() # Explicitly rerun to refresh the background with the new manual shape

        st.divider()
        st.header("Generate Final Document")
        if st.button("Generate Redacted PDF"):
            with st.spinner("Applying all redactions and removals..."):
                approved_areas_by_page = defaultdict(list)

                # Add AI-approved redactions (these are already in PDF point coordinates)
                approved_suggestions = [s for s in st.session_state.suggestions if st.session_state.approval_state.get(s['id'])]
                for s in approved_suggestions:
                    approved_areas_by_page[s['page_num']].extend(s.get('rects', []))

                # Add duplicate email removals (these are also in PDF point coordinates)
                approved_duplicate_emails = [d for d in st.session_state.duplicate_emails if st.session_state.duplicate_email_state.get(d['id'])]
                for d in approved_duplicate_emails:
                    approved_areas_by_page[d['page_num']].extend(d.get('rects', []))

                # Add manually drawn redactions, scaling them correctly back to PDF coordinates
                for page_num, canvas_objects in st.session_state.manual_rects.items():
                    original_img_width = st.session_state.original_pdf_images[page_num].width
                    # This factor converts from the displayed canvas coordinates back to PDF points
                    final_scaling_factor = (72.0 / PREVIEW_DPI) * (original_img_width / CANVAS_DISPLAY_WIDTH)
                    for obj in canvas_objects:
                        x1, y1 = obj["left"], obj["top"]
                        x2, y2 = x1 + obj["width"], y1 + obj["height"]
                        pdf_rect = fitz.Rect(
                            x1 * final_scaling_factor, y1 * final_scaling_factor,
                            x2 * final_scaling_factor, y2 * final_scaling_factor
                        )
                        approved_areas_by_page[page_num].append(pdf_rect)

                if not approved_areas_by_page:
                    st.warning("No redactions or removals were selected or drawn.")
                else:
                    final_redaction_areas = list(approved_areas_by_page.items())
                    output_filename = os.path.splitext(os.path.basename(st.session_state.processed_file))[0] + "_redacted.pdf"
                    output_pdf_path = os.path.join(output_dir, output_filename)

                    processor = PDFProcessor(st.session_state.processed_file)
                    processor.apply_redactions(final_redaction_areas, output_pdf_path)

                    st.session_state.final_pdf_path = output_pdf_path
                    st.success(f"Successfully created redacted document: {output_filename}")

    if st.session_state.final_pdf_path and os.path.exists(st.session_state.final_pdf_path):
        with open(st.session_state.final_pdf_path, "rb") as f:
            st.download_button(
                "Download Redacted PDF", 
                f, 
                file_name=os.path.basename(st.session_state.final_pdf_path), 
                mime="application/pdf"
            )



if __name__ == "__main__":
    main()  