import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentParagraph, DocumentSpan
import re
from fuzzywuzzy import fuzz
import fitz
import os
from PIL import Image
from io import BytesIO
import csv
from datetime import datetime
from pathlib import Path
import difflib

# --- Logger Setup ---
def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()

# --- Rectangle Merging Logic ---
def merge_consecutive_word_rects(word_rects: List[fitz.Rect]) -> List[fitz.Rect]:
    if not word_rects:
        return []
    lines = defaultdict(list)
    for rect in word_rects:
        lines[round(rect.y0)].append(rect)
    final_merged_rects = []
    for line_y in sorted(lines.keys()):
        sorted_rects = sorted(lines[line_y], key=lambda r: r.x0)
        if not sorted_rects:
            continue
        current_run_group = [sorted_rects[0]]
        for i in range(1, len(sorted_rects)):
            prev_rect = current_run_group[-1]
            current_rect = sorted_rects[i]
            max_gap = prev_rect.height * 0.75 
            actual_gap = current_rect.x0 - prev_rect.x1
            if actual_gap <= max_gap:
                current_run_group.append(current_rect)
            else:
                merged_run = fitz.Rect()
                for r in current_run_group:
                    merged_run |= r
                final_merged_rects.append(merged_run)
                current_run_group = [current_rect]
        if current_run_group:
            merged_run = fitz.Rect()
            for r in current_run_group:
                merged_run |= r
            final_merged_rects.append(merged_run)
    return final_merged_rects

# --- Mapping LLM Findings to Document Coordinates ---
def create_detailed_suggestions(
    analysis: AnalyzeResult, 
    all_findings_with_source: List[Dict]
) -> List[Dict]:
    """
    Maps sensitive text from an LLM to its specific coordinates, handling sources that are either
    a specific paragraph (for PII) or a whole page (for sensitive content).
    """
    detailed_suggestions = []
    scaling_factor = 72.0
    
    # --- Create a map of all words on each page, marked as "unused" ---
    words_by_page = defaultdict(list)
    for page in analysis.pages:
        for word in page.words:
            words_by_page[page.page_number - 1].append({'word_obj': word, 'used': False})

    suggestion_id_counter = 0

    # We sort all findings by length to prioritise longer matches across the board
    sorted_findings = sorted(all_findings_with_source, key=lambda x: len(x['llm_finding']['text']), reverse=True)
        
    for item in sorted_findings:
        llm_finding = item['llm_finding']
        text_to_find = llm_finding['text']
        source_paragraph = item.get('source_paragraph')
        source_page = item.get('source_page')
        words_to_search = []
        context = ""
        page_num = -1

        if source_paragraph:
            page_num = source_paragraph.bounding_regions[0].page_number - 1
                    
            para_span = source_paragraph.spans[0]
            words_to_search = [
                w_dict for w_dict in words_by_page.get(page_num, [])
                if w_dict['word_obj'].span.offset >= para_span.offset and
                   (w_dict['word_obj'].span.offset + w_dict['word_obj'].span.length) <= (para_span.offset + para_span.length)
            ]
            context = source_paragraph.content

        elif source_page:
            page_num = source_page.page_number - 1
            words_to_search = words_by_page.get(page_num, [])
            context = analysis.content[source_page.spans[0].offset : source_page.spans[0].offset + source_page.spans[0].length]

        else:
            continue  # Skip if neither source_paragraph nor source_page is available

        # Normalization for matching
        norm_text_to_find = text_to_find.lower().replace("’s", "").replace("'s", "").replace("'", "").replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
        
        best_match_words_info = []
        best_match_score = 0

        for i in range(len(words_to_search)):
            for j in range(i, len(words_to_search)):
                candidate_word_info = words_to_search[i:j+1]
                
                # Skip if any word in this candidate sequence is already used
                if any(w['used'] for w in candidate_word_info):
                    continue

                candidate_words = [w['word_obj'] for w in candidate_word_info]
                reconstructed_text = "".join([w.content for w in candidate_words])
                norm_reconstructed = reconstructed_text.lower().replace("’s", "").replace("'s", "").replace("'", "").replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(" ", "").replace('"', '').replace('“', '').replace('”', '')
                
                score = fuzz.ratio(norm_reconstructed, norm_text_to_find)
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_words_info = candidate_word_info
                
                if best_match_score == 100: break
            if best_match_score == 100: break
        
        if best_match_score >= 90 and best_match_words_info:
            # Mark these words as "used" so they can't be matched again
            for w_info in best_match_words_info:
                w_info['used'] = True

            best_match_words = [w_info['word_obj'] for w_info in best_match_words_info]
            
            individual_word_rects = []
            for word_obj in best_match_words:
                if word_obj.polygon and len(word_obj.polygon) >= 8:
                    points = [
                        fitz.Point(word_obj.polygon[k] * scaling_factor, word_obj.polygon[k+1] * scaling_factor) 
                        for k in range(0, len(word_obj.polygon), 2)
                    ]
                    individual_word_rects.append(fitz.Quad(points).rect)
            
            if individual_word_rects:
                merged_line_rects = merge_consecutive_word_rects(individual_word_rects)
                detailed_suggestions.append({
                    'id': suggestion_id_counter, 'text': llm_finding['text'], 'category': llm_finding['category'],
                    'reasoning': llm_finding['reasoning'], 'context': context,
                    'page_num': page_num, 'rects': merged_line_rects
                })
                suggestion_id_counter += 1
    
    logger.info(f"Successfully created {len(detailed_suggestions)} detailed suggestions from {len(all_findings_with_source)} LLM findings.")
    if len(detailed_suggestions) != len(all_findings_with_source):
        logger.warning(f"Could not find a unique physical location for {len(all_findings_with_source) - len(detailed_suggestions)} LLM findings.")
        
        # Use Counter to correctly handle duplicate text entries
        llm_text_counts = Counter(item['llm_finding']['text'] for item in all_findings_with_source)
        mapped_text_counts = Counter(s['text'] for s in detailed_suggestions)
        
        # Subtract the mapped counts from the LLM counts to find the difference
        unmapped = llm_text_counts - mapped_text_counts
        
        if unmapped:
            logger.error("--- MISMATCH REPORT: The following LLM findings could not be located in the document ---")
            for text, count in unmapped.items():
                logger.error(f"  - Text: '{text}' (LLM found {count} unmapped instance(s))")
            logger.error("--- END OF REPORT ---")

    detailed_suggestions = sorted(
        detailed_suggestions, 
        key=lambda s: (s['page_num'], s['rects'][0].y0 if s['rects'] else 0)
    )
            
    return detailed_suggestions

# --- PDF to Image Conversion for Preview ---
PREVIEW_DPI = 150
def get_original_pdf_images(pdf_path):
    """Extracts each page of a PDF as a Pillow Image object."""
    if not os.path.exists(pdf_path): return []
    try:
        doc = fitz.open(pdf_path)
        images = [Image.open(BytesIO(page.get_pixmap(dpi=PREVIEW_DPI).tobytes("png"))) for page in doc]
        doc.close()
        return images
    except Exception as e:
        logger.error(f"Error opening or rendering PDF: {e}")
        return []



# --- Logger that saves output of NER calls (for testing/evaluating) ---

def log_ner_output(base_filename, *values, header=None, sep=","):
    """
    Creates a new log file with a timestamp in its name on the first call,
    then appends rows to that file on subsequent calls.

    Args:
        base_filename (str): Base name for the log file (e.g., 'training_log').
        *values: Values to log as a single row.
        header (list[str] | None): Optional header written only on the first call.
        sep (str): Delimiter for the CSV file. Defaults to ",".

    Example:
        for i in range(3):
            loss = 0.1 * i
            acc = 0.8 + 0.05 * i
            log_ner_output("training_log", i, loss, acc, header=["epoch", "loss", "acc"])
    """
    # Initialize state tracking
    if not hasattr(log_ner_output, "_log_file"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_ner_output._log_file = Path(f"logs/{base_filename}_{timestamp}.csv")
        log_ner_output._initialized = False

    # Choose mode: overwrite on first call, append otherwise
    mode = "w" if not log_ner_output._initialized else "a"

    with log_ner_output._log_file.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=sep)

        # Write header only once, if provided
        if not log_ner_output._initialized and header:
            writer.writerow(header)

        # Write the row
        writer.writerow(values)

    # Mark as initialized
    log_ner_output._initialized = True



# --- Extract Emails from AnalyzeResult ---

def extract_emails_alt(analyze_result):
    # Usage:
    # analyze_result = client.begin_analyze_document("prebuilt-layout", document).result()
    # email_paragraphs = extract_emails_from_analyze_result(analyze_result)
    # print(email_paragraphs)

    raw_text = analyze_result.content

    # Regex to find emails
    email_pattern = re.compile(
        r"From:\s*(?P<from>.+?)\s*To:\s*(?P<to>.+?)\s*Subject:\s*(?P<subject>.+?)\s*Date:\s*(?P<date>.+?)\s*(?P<body>(?:.|\n)*?)(?=From:|$)",
        re.MULTILINE
    )

    email_paragraphs = []

    for match in email_pattern.finditer(raw_text):
        start_offset = match.start()
        end_offset = match.end()
        email_text = raw_text[start_offset:end_offset]

        # Collect bounding regions across pages
        merged_points = []
        for page in analyze_result.pages:
            for region in page.get("boundingRegions", []):
                span_offset = region["span"]["offset"]
                if start_offset <= span_offset < end_offset:
                    merged_points.extend(region["polygon"])

        # Merge bounding box if points exist
        merged_region = None
        if merged_points:
            xs = [p["x"] for p in merged_points]
            ys = [p["y"] for p in merged_points]
            merged_polygon = [
                {"x": min(xs), "y": min(ys)},
                {"x": max(xs), "y": min(ys)},
                {"x": max(xs), "y": max(ys)},
                {"x": min(xs), "y": max(ys)}
            ]
            merged_region = {"pageNumber": None, "polygon": merged_polygon}

        # Build custom paragraph-like object
        paragraph = {
            "content": email_text,
            "role": "email",
            "span": {"offset": start_offset, "length": end_offset - start_offset},
            "bounding_region": merged_region
        }
        email_paragraphs.append(paragraph)

    return email_paragraphs



# Regex to find email elements
# Flags used across all patterns
RE_FLAGS = re.IGNORECASE | re.MULTILINE

# 1) Generic header line with folding support:
#    - Captures the value on the same line and any subsequent continuation lines (starting with space or tab).
HDR_VALUE = r"[^\r\n]*?(?:\r?\n[ \t].*?)*"

# 2) Core headers
RE_FROM    = re.compile(rf"^From:\s*(?P<from>{HDR_VALUE})\s*$", RE_FLAGS)
RE_TO      = re.compile(rf"^To:\s*(?P<to>{HDR_VALUE})\s*$", RE_FLAGS)
RE_CC      = re.compile(rf"^Cc:\s*(?P<cc>{HDR_VALUE})\s*$", RE_FLAGS)
RE_BCC     = re.compile(rf"^Bcc:\s*(?P<bcc>{HDR_VALUE})\s*$", RE_FLAGS)
RE_SUBJECT = re.compile(rf"^Subject:\s*(?P<subject>{HDR_VALUE})\s*$", RE_FLAGS)
RE_DATE    = re.compile(rf"^(?:Date|Sent):\s*(?P<date>{HDR_VALUE})\s*$", RE_FLAGS)

# 3) Optional but useful headers
RE_MESSAGE_ID   = re.compile(rf"^Message-Id:\s*(?P<message_id>{HDR_VALUE})\s*$", RE_FLAGS)
RE_IN_REPLY_TO  = re.compile(rf"^In-Reply-To:\s*(?P<in_reply_to>{HDR_VALUE})\s*$", RE_FLAGS)
RE_REFERENCES   = re.compile(rf"^References:\s*(?P<references>{HDR_VALUE})\s*$", RE_FLAGS)
RE_CONTENT_TYPE = re.compile(rf"^Content-Type:\s*(?P<content_type>{HDR_VALUE})\s*$", RE_FLAGS)

# 4) Email address and name extraction (for post-processing values)
#    - Matches: "Name <user@domain>", "user@domain", '"Quoted Name" <user@domain>'
RE_NAME_EMAIL = re.compile(
    r"""
    (?P<name>"[^"]+"|[^<>"\r\n]+?)?            # optional display name (quoted or unquoted)
    \s*<?\s*
    (?P<email>[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})  # email
    \s*>?
    """,
    re.IGNORECASE | re.VERBOSE
)

# 5) Thread quote markers (to segment quoted blocks within bodies)
RE_QUOTED_WROTE = re.compile(
    r"""^On\s+.+?,\s+.+?\s+wrote:\s*$""",
    re.IGNORECASE | re.MULTILINE
)
RE_ORIGINAL_MESSAGE = re.compile(
    r"""^-{2,}\s*Original Message\s*-{2,}\s*$""",
    re.IGNORECASE | re.MULTILINE
)


def unfold_headers(header_block: str) -> str:
    """
    RFC 5322 header unfolding: join lines where continuation starts with space/tab.
    (We keep CR/LF consistent to not break upstream spans mapping.)
    """
    lines = header_block.splitlines(keepends=True)
    out = []
    for line in lines:
        if out and (line.startswith(" ") or line.startswith("\t")):
            out[-1] = out[-1].rstrip("\r\n") + " " + line.lstrip()
        else:
            out.append(line)
    return "".join(out)

def find_header_body_split(text: str) -> Optional[int]:
    """
    Returns the index in `text` where the headers end and the body begins:
    the first occurrence of a blank line (\n\n or \r\n\r\n).
    """
    m = re.search(r"\r?\n\r?\n", text)
    return m.end() if m else None

def extract_first(pattern: re.Pattern, text: str, group: str) -> Optional[str]:
    m = pattern.search(text)
    if not m:
        return None
    return m.group(group).strip()

def parse_header_block(header_block: str) -> Dict[str, Any]:
    """
    Parse unfolded header block into a dict of values (raw strings).
    """
    h = {}
    h["from"]       = extract_first(RE_FROM, header_block, "from")
    h["to"]         = extract_first(RE_TO, header_block, "to")
    h["cc"]         = extract_first(RE_CC, header_block, "cc")
    h["bcc"]        = extract_first(RE_BCC, header_block, "bcc")
    h["subject"]    = extract_first(RE_SUBJECT, header_block, "subject")
    h["date"]       = extract_first(RE_DATE, header_block, "date")
    h["message_id"] = extract_first(RE_MESSAGE_ID, header_block, "message_id")
    h["in_reply_to"]= extract_first(RE_IN_REPLY_TO, header_block, "in_reply_to")
    h["references"] = extract_first(RE_REFERENCES, header_block, "references")
    h["content_type"]=extract_first(RE_CONTENT_TYPE, header_block, "content_type")
    return h

def split_recipients(raw: Optional[str]) -> List[Dict[str, Optional[str]]]:
    """
    Split a To/Cc/Bcc header into a list of {name, email} dicts.
    Handles commas and semicolons as separators and optional display names.
    """
    if not raw:
        return []
    parts = re.split(r";,*$)", raw)  # split by commas/semicolons ignoring commas inside quotes
    recipients = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = RE_NAME_EMAIL.search(part)
        if m:
            name = m.group("name")
            if name:
                name = name.strip().strip('"')
            recipients.append({"name": name or None, "email": m.group("email")})
        else:
            # fallback: no email matched, keep as raw
            recipients.append({"name": None, "email": part})
    return recipients



def parse_single_email_block(block_text: str, block_global_offset: int) -> Dict[str, Any]:
    """
    Parse ONE email block consisting of headers + blank line + body.
    Returns structured fields and span offsets.
    `block_global_offset` is the starting offset of block_text in the original document text.
    """
    split_idx = find_header_body_split(block_text)
    if split_idx is None:
        # No clear blank line: treat entire block as body-only with unknown headers
        headers_unfolded = ""
        body = block_text.strip()
        header_span = None
        body_span = {"offset": block_global_offset, "length": len(block_text)}
    else:
        headers_raw = block_text[:split_idx]
        headers_unfolded = unfold_headers(headers_raw)
        body = block_text[split_idx:].lstrip("\r\n")  # remove the separating blank line
        header_span = {"offset": block_global_offset, "length": len(headers_raw)}
        body_start = block_global_offset + split_idx
        # account for lstrip
        leading = len(block_text[split_idx:]) - len(body)
        body_span = {"offset": body_start + (0), "length": len(body)}

    header_fields = parse_header_block(headers_unfolded)

    # Parse recipient lists
    to_list  = split_recipients(header_fields.get("to"))
    cc_list  = split_recipients(header_fields.get("cc"))
    bcc_list = split_recipients(header_fields.get("bcc"))

    return {
        "full_text": block_text,
        "from_raw": header_fields.get("from"),
        "to_raw": header_fields.get("to"),
        "cc_raw": header_fields.get("cc"),
        "bcc_raw": header_fields.get("bcc"),
        "subject": header_fields.get("subject"),
        "date": header_fields.get("date"),
        "message_id": header_fields.get("message_id"),
        "in_reply_to": header_fields.get("in_reply_to"),
        "references": header_fields.get("references"),
        "content_type": header_fields.get("content_type"),
        "to": to_list,
        "cc": cc_list,
        "bcc": bcc_list,
        "body": body.strip("\r\n"),
        "spans": {
            "email": {"offset": block_global_offset, "length": len(block_text)},
            "headers": header_span,
            "body": body_span,
        }
    }




import re

def parse_email_block(block_text, global_offset):
    """
    Parse a single email block into headers and body.
    Returns a dict with headers, body, and spans.
    """
    lines = block_text.splitlines()
    headers = {}
    body_lines = []
    in_headers = True

    header_patterns = {
        "from": re.compile(r"^From:\s*(.+)", re.IGNORECASE),
        "to": re.compile(r"^To:\s*(.+)", re.IGNORECASE),
        "cc": re.compile(r"^CC:\s*(.+)", re.IGNORECASE),
        "bcc": re.compile(r"^BCC:\s*(.+)", re.IGNORECASE),
        "subject": re.compile(r"^Subject:\s*(.+)", re.IGNORECASE),
        "date": re.compile(r"^Date:\s*(.+)", re.IGNORECASE),
    }

    current_offset = global_offset
    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if in_headers:
            matched = False
            for key, pattern in header_patterns.items():
                m = pattern.match(line)
                if m:
                    headers[key] = m.group(1).strip()
                    matched = True
                    break
            if not matched and line.strip() != "":
                # First non-header line → start body
                in_headers = False
                body_lines.append(line)
        else:
            body_lines.append(line)
        current_offset += line_len

    body_text = "\n".join(body_lines).strip()
    return {
        "headers": headers,
        "body": body_text,
        "span": {"offset": global_offset, "length": len(block_text)}
    }



def split_thread_into_email_blocks(full_text: str):
    """
    Splits stacked email threads into blocks by detecting header starts or quoted markers.
    Handles cases where middle emails only have 'On ... wrote:' markers.
    """
    text = full_text.strip()

    # Regex: detect new email boundaries
    marker = re.compile(
        r"(?im)(?=^\s*From:|^\s*-{2,}\s*Original Message\s*-{2,}|^\s*On\s.+?\s+wrote:)",
        re.MULTILINE
    )

    indices = [m.start() for m in marker.finditer(text)]
    if not indices:
        return [(0, text)]

    blocks = []
    for idx, start in enumerate(indices):
        end = indices[idx + 1] if idx + 1 < len(indices) else len(text)
        blocks.append((start, text[start:end].strip()))
    return blocks



def parse_thread(full_text: str) -> List[Dict[str, Any]]:
    blocks = split_thread_into_email_blocks(full_text)
    emails = []
    for start, block in blocks:
        # emails.append(parse_single_email_block(block, start))
        emails.append(parse_email_block(block, start))
    return emails




def extract_emails(analyze_result):
    raw_text = analyze_result.content
    lines = raw_text.splitlines()

    emails = []
    current_email = {}
    body_lines = []
    in_body = False
    email_start_offset = None

    # Track current offset as we iterate through lines
    current_offset = 0

    for line in lines:
        line_stripped = line.strip()
        line_length = len(line) + 1  # +1 for newline

        # Detect start of a new email
        if line_stripped.startswith("From:"):
            # Save previous email if exists
            if current_email:
                email_end_offset = current_offset
                current_email["body"] = "\n".join(body_lines).strip()

                # Merge bounding boxes for this email
                merged_region = merge_bounding_boxes(analyze_result, email_start_offset, email_end_offset)

                email_tmp = {
                    "from": current_email.get("from"),
                    "to": current_email.get("to"),
                    "subject": current_email.get("subject"),
                    "date": current_email.get("date"),
                    "body": current_email.get("body"),
                    "content": raw_text[email_start_offset:email_end_offset],
                    "span": {"offset": email_start_offset, "length": email_end_offset - email_start_offset},
                    "bounding_region": merged_region
                }

                emails.append(get_body_span(email_tmp, raw_text))

                current_email = {}
                body_lines = []
                in_body = False

            # Start new email
            email_start_offset = current_offset
            current_email["from"] = line_stripped.replace("From:", "").strip()
            current_offset += line_length
            continue

        # Collect headers
        if not in_body:
            if line_stripped.startswith("To:"):
                current_email["to"] = line_stripped.replace("To:", "").strip()
            elif line_stripped.startswith("Subject:"):
                current_email["subject"] = line_stripped.replace("Subject:", "").strip()
            elif line_stripped.startswith("Date:"):
                current_email["date"] = line_stripped.replace("Date:", "").strip()
            elif line_stripped == "":
                # Blank line → start of body
                in_body = True
            current_offset += line_length
            continue

        # Collect body lines
        if in_body:
            body_lines.append(line)
            current_offset += line_length

    # Save last email
    if current_email:
        email_end_offset = current_offset
        current_email["body"] = "\n".join(body_lines).strip()
        merged_region = merge_bounding_boxes(analyze_result, email_start_offset, email_end_offset)

        email_tmp = {
            "from": current_email.get("from"),
            "to": current_email.get("to"),
            "subject": current_email.get("subject"),
            "date": current_email.get("date"),
            "body": current_email.get("body"),
            "content": raw_text[email_start_offset:email_end_offset],
            "span": {"offset": email_start_offset, "length": email_end_offset - email_start_offset},
            "bounding_region": merged_region
        }

        emails.append(get_body_span(email_tmp, raw_text))

    return emails





def merge_bounding_boxes(analyze_result, start_offset, end_offset):
    """Merge bounding boxes for all lines that intersect with the email span."""
    all_points = []

    for page in analyze_result.pages:
        for line in page.lines or []:
            for span in line.spans:
                if span.offset >= start_offset and span.offset < end_offset:
                    # line.polygon is a flat list: [x1, y1, x2, y2, ...]
                    coords = list(zip(line.polygon[0::2], line.polygon[1::2]))
                    all_points.extend(coords)

    if not all_points:
        return None

    xs = [x for x, y in all_points]
    ys = [y for x, y in all_points]

    merged_polygon = [
        {"x": min(xs), "y": min(ys)},
        {"x": max(xs), "y": min(ys)},
        {"x": max(xs), "y": max(ys)},
        {"x": min(xs), "y": max(ys)}
    ]
    return {"pageNumber": None, "polygon": merged_polygon}






def normalize_with_map(s: str) -> Tuple[str, List[int]]:
    """
    Lowercase + collapse all whitespace to single spaces.
    Returns:
      norm: normalized string
      idx_map: list mapping each char index in norm -> original index in s
    """
    norm_chars = []
    idx_map = []

    i = 0
    last_was_space = False
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            if not last_was_space:
                norm_chars.append(' ')
                idx_map.append(i)  # map space to the first whitespace index
                last_was_space = True
        else:
            norm_chars.append(ch.lower())
            idx_map.append(i)
            last_was_space = False
        i += 1

    # Don’t strip to avoid remapping; instead, trim logical search windows later if needed.
    norm = ''.join(norm_chars)
    return norm, idx_map



def get_body_span(email: Dict[str, Any], analyze_result_content: str) -> Dict[str, Any]:
    """
    Ensures email contains a 'body_span' (global offsets).
    It searches the email.body relative to email.span content region.
    """
    if "body_span" in email and email["body_span"]:
        return email

    body = email.get("body", "")
    email_span = email.get("span") or {}
    email_start = email_span.get("offset", None)
    email_len = email_span.get("length", None)

    if email_start is None or email_len is None or not body:
        email["body_span"] = None
        return email

    email_end = email_start + email_len
    email_text_global = analyze_result_content[email_start:email_end]

    # Find body inside the email block (exact fallback search)
    idx = email_text_global.find(body)
    if idx == -1:
        # Very robust fallback: collapse whitespace for both and map back.
        email_norm, email_map = normalize_with_map(email_text_global)
        body_norm, _ = normalize_with_map(body)
        idx_norm = email_norm.find(body_norm)
        if idx_norm == -1:
            email["body_span"] = None
        else:
            start_in_email = email_map[idx_norm]
            end_in_email = email_map[min(idx_norm + len(body_norm) - 1, len(email_map) - 1)] + 1
            email["body_span"] = {"offset": email_start + start_in_email, "length": end_in_email - start_in_email}
    else:
        email["body_span"] = {"offset": email_start + idx, "length": len(body)}

    return email



def per_page_boxes_for_span(analyze_result, start_offset: int, end_offset: int) -> Dict[int, Dict[str, float]]:
    """
    Returns merged rectangles per page that intersect [start_offset, end_offset).
    Output format:
      { pageNumber: {"x1":..., "y1":..., "x2":..., "y2":...}, ... }
    Coordinates are in the same units as Document Intelligence polygons.
    """
    by_page_points = {}  # page_num -> list[(x,y)]

    for page in analyze_result.pages:
        if not getattr(page, "lines", None):
            continue
        for line in page.lines:
            # Some SDKs expose line.spans; if absent, skip positional mapping for this line
            if not getattr(line, "spans", None):
                continue
            # If any of the line’s spans intersects the target range, include its polygon
            if any((span.offset < end_offset) and (span.offset + span.length > start_offset) for span in line.spans):
                # line.polygon is a flat list: [x1, y1, x2, y2, ...]
                poly = getattr(line, "polygon", None)
                if not poly:
                    continue
                coords = list(zip(poly[0::2], poly[1::2]))
                by_page_points.setdefault(page.page_number, []).extend(coords)

    per_page_boxes = {}
    for page_num, pts in by_page_points.items():
        xs = [x for x, y in pts]
        ys = [y for x, y in pts]
        per_page_boxes[page_num] = {
            "x1": min(xs), "y1": min(ys),  # top-left
            "x2": max(xs), "y2": max(ys),  # bottom-right
        }
    return per_page_boxes




def find_stacked_email_duplicates(
    emails: List[Dict[str, Any]],
    analyze_result
) -> List[Dict[str, Any]]:
    """
    Detects content from earlier emails that is included (quoted) in later emails.
    Returns a list of duplicate segments with global spans and per-page boxes.
    """
    content = analyze_result.content

    # Ensure each email has a body_span
    for e in emails:
        ensure_body_span(e, content)

    # Precompute normalized bodies and maps
    body_norms = []
    for e in emails:
        body = e.get("body") or ""
        norm, idx_map = normalize_with_map(body)
        body_norms.append({"norm": norm, "map": idx_map, "raw": body})

    results = []

    # Compare each later email (i) against each prior (j)
    for i in range(len(emails)):
        ei = emails[i]
        ei_body_span = ei.get("body_span")
        if not ei_body_span:
            continue
        ei_body_start = ei_body_span["offset"]
        ei_body_len = ei_body_span["length"]
        ei_body_end = ei_body_start + ei_body_len
        ei_body = content[ei_body_start:ei_body_end]

        ei_norm, ei_map = normalize_with_map(ei_body)

        for j in range(i):  # prior emails are potential sources of quoted text
            ej = emails[j]
            ej_body = ej.get("body") or ""
            if not ej_body.strip():
                continue
            ej_norm = body_norms[j]["norm"]

            # ---- Exact inclusion (normalized) ----
            start_idx = 0
            found_any = False
            while True:
                k = ei_norm.find(ej_norm, start_idx)
                if k == -1:
                    break
                found_any = True

                # Map normalized indices back to original global offsets
                dup_start_in_ei = ei_map[k]
                dup_end_in_ei = ei_map[min(k + len(ej_norm) - 1, len(ei_map) - 1)] + 1
                dup_global_start = ei_body_start + dup_start_in_ei
                dup_global_end = ei_body_start + dup_end_in_ei

                # Per-page boxes
                per_page = per_page_boxes_for_span(analyze_result, dup_global_start, dup_global_end)

                results.append({
                    "source_email_index": j,
                    "target_email_index": i,
                    "type": "exact",
                    "span": {"offset": dup_global_start, "length": dup_global_end - dup_global_start},
                    "per_page_boxes": per_page,
                })
                start_idx = k + len(ej_norm)

            if found_any:
                continue  # don’t fuzzy-match if exact inclusion already found

            # ---- Fuzzy fallback (optional) ----
            # If there are small edits, find a large matching region.
            # Tune 'coverage_threshold' to how strict you want this to be.
            coverage_threshold = 0.8  # 80% of prior email body must appear
            sm = difflib.SequenceMatcher(None, ei_norm, ej_norm)
            blocks = sm.get_matching_blocks()

            # Merge contiguous blocks in ei_norm
            if blocks:
                i_spans = []
                covered = 0
                for b in blocks:
                    if b.size == 0:
                        continue
                    i_spans.append((b.a, b.a + b.size))
                    covered += b.size

                if covered >= coverage_threshold * max(1, len(ej_norm)):
                    # Merge to a single continuous range (loose visual bounding)
                    i_start = min(a for a, b in i_spans)
                    i_end = max(b for a, b in i_spans)

                    dup_start_in_ei = ei_map[i_start]
                    dup_end_in_ei = ei_map[min(i_end - 1, len(ei_map) - 1)] + 1
                    dup_global_start = ei_body_start + dup_start_in_ei
                    dup_global_end = ei_body_start + dup_end_in_ei

                    per_page = per_page_boxes_for_span(analyze_result, dup_global_start, dup_global_end)
                    results.append({
                        "source_email_index": j,
                        "target_email_index": i,
                        "type": "fuzzy",
                        "coverage": covered / max(1, len(ej_norm)),
                        "span": {"offset": dup_global_start, "length": dup_global_end - dup_global_start},
                        "per_page_boxes": per_page,
                    })

    return results




import re
import difflib
from typing import List, Dict, Any, Optional

def find_duplicate_emails(
    emails: List[Dict[str, Any]],
    text_key: str = "body",
    min_chars: int = 30,
    fuzzy: bool = True,
    fuzzy_coverage: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Identify duplicates where later emails include the content of earlier emails (quoted replies).
    
    Args:
        emails: list of dicts; each dict must contain the email text under `text_key`.
                You can store anything else in the dict (From, Subject, etc.)—the function ignores it.
        text_key: the key that contains the email content (e.g., "body" or "content").
        min_chars: minimum length (after normalization) to consider a candidate as duplicate.
        fuzzy: if True, also detect near-duplicates (≥ fuzzy_coverage of prior email appears).
        fuzzy_coverage: required fraction (0-1) of the prior email's normalized text that must match.

    Returns:
        A list of dicts, each with:
          - source_index: index of the earlier email whose content was quoted
          - target_index: index of the later email that contains the quote
          - type: "exact" or "fuzzy"
          - content: the duplicated text snippet (from the source email)
          - target_span: (start, end) char indices within the target email's original text
                         (based on `text_key` string)
          - coverage (for fuzzy): fraction of source matched in target
    """
    # --- helpers ---
    def normalize(s: str) -> str:
        # Lowercase + collapse whitespace; keep it simple and stable
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def index_in_original(norm_target: str, orig_target: str, norm_source: str, orig_source: str) -> Optional[tuple]:
        """
        Map the normalized match back to indices in the original target string.
        We use a simplistic approach: find normalized source in normalized target,
        then locate the first and last non-space characters of that slice back in the original.
        """
        k = norm_target.find(norm_source)
        if k == -1:
            return None

        # Build a naive mapping from normalized indices to original indices by scanning
        # This avoids heavy bookkeeping and works well in practice for email bodies.
        def build_map(s: str):
            m = []
            i = 0
            j = 0
            while i < len(s):
                if s[i].isspace():
                    # collapse to single space in normalized; map that space to first whitespace char
                    while i < len(s) and s[i].isspace():
                        if len(m) == 0 or m[-1] != i:
                            pass
                        i += 1
                    j += 1  # one normalized char for any whitespace run
                else:
                    i += 1
                    j += 1
            # We don't return the map here to keep it lightweight; we’ll fallback to substring find
            # in original where possible.
            return

        # Simple fallback: try to locate the original source text directly (case-insensitive)
        low_target = orig_target.lower()
        low_source = orig_source.lower()
        k_orig = low_target.find(low_source)
        if k_orig != -1:
            return (k_orig, k_orig + len(orig_source))

        # If that failed (e.g., whitespace differences), approximate by expanding around k
        # Count normalized chars to original chars approximately by scanning orig_target
        # and building a mapping only for the needed window.
        # Create normalized with pointers to original indices
        n_chars = []
        n_to_o = []
        i = 0
        last_space = False
        while i < len(orig_target):
            ch = orig_target[i]
            if ch.isspace():
                if not last_space:
                    n_chars.append(' ')
                    n_to_o.append(i)
                    last_space = True
                # skip remaining whitespace in a run
            else:
                n_chars.append(ch.lower())
                n_to_o.append(i)
                last_space = False
            i += 1

        # We now have a full map; derive original indices for the normalized window
        if k < len(n_to_o):
            start_o = n_to_o[k]
            end_norm_idx = min(k + len(norm_source) - 1, len(n_to_o) - 1)
            end_o = n_to_o[end_norm_idx] + 1
            return (start_o, end_o)
        return None

    duplicates = []

    # Precompute normalized texts
    norm_texts = [normalize(e.get(text_key, "")) for e in emails]
    orig_texts = [e.get(text_key, "") for e in emails]

    # Compare each later email i against each earlier email j
    for i in range(len(emails)):
        tgt_norm = norm_texts[i]
        tgt_orig = orig_texts[i]
        if not tgt_norm:
            continue

        for j in range(i):
            src_norm = norm_texts[j]
            src_orig = orig_texts[j]
            if not src_norm or len(src_norm) < min_chars:
                continue

            # ---- exact (normalized substring) ----
            k = tgt_norm.find(src_norm)
            if k != -1:
                span = index_in_original(tgt_norm, tgt_orig, src_norm, src_orig)
                duplicates.append({
                    "source_email_index": j,
                    "target_email_index": i,
                    "type": "exact",
                    "content": emails[j].get(text_key, ""),
                    "target_span": span if span else (0, 0),
                })
                continue  # already exact; skip fuzzy

            # ---- fuzzy (optional) ----
            if fuzzy:
                sm = difflib.SequenceMatcher(None, tgt_norm, src_norm)
                blocks = sm.get_matching_blocks()
                covered = sum(b.size for b in blocks if b.size > 0)
                coverage = covered / max(1, len(src_norm))
                if coverage >= fuzzy_coverage:
                    # Use the min/max 'a' indices in target (tgt_norm) to build a span
                    a_positions = [(b.a, b.a + b.size) for b in blocks if b.size > 0]
                    # Merge to a single window (keeps it "maximally simple")
                    a_start = min(a for a, b in a_positions)
                    a_end = max(b for a, b in a_positions)
                    # Map the normalized window back to original indices
                    window_norm = tgt_norm[a_start:a_end]
                    span = index_in_original(tgt_norm, tgt_orig, window_norm, tgt_orig[a_start:a_end])
                    duplicates.append({
                        "source_email_index": j,
                        "target_email_index": i,
                        "type": "fuzzy",
                        "coverage": round(coverage, 3),
                        "content": emails[j].get(text_key, ""),
                        "target_span": span if span else (0, 0),
                    })

    return duplicates
