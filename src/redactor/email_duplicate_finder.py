from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from azure.ai.documentintelligence.models import AnalyzeResult
import re
from fuzzywuzzy import fuzz
import fitz
import difflib

# -------- Patterns for matching emails and their sections (e.g. headers, body) --------
# Generic header line: "<name> : <value>" (name: letters, spaces, hyphens)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b")
HEADER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z\- ]{0,40}?)\s*:\s*(.+)$")
ORIGINAL_MSG_RE = re.compile(r"^-{2,}\s*original message\s*-{2,}$", re.I)
ON_WROTE_RE = re.compile(r"^\s*on .{0,200}wrote:?\s*$", re.I)
# Marker that delineates start of an email block. Here, email blocks can start with 'From:' or 'On ... wrote:'
# EMAIL_START_BOUNDARY = re.compile(r"(?i)(?=From:\s*[A-Z]|On\s+.+?\s+wrote:)", re.MULTILINE)
# EMAIL_START_BOUNDARY = re.compile(r"(?m)(?=^((?:>{0,10}\s*)?On .+?wrote: | From:\s.+ | -----Original Message----- | -{2,}\s*Forwarded message\s*-{2,})$)")
EMAIL_START_BOUNDARY = re.compile(r"(?i)(?=From:\s*[A-Z]|On\s+.+?\s+wrote:|-{2,}\s*Original Message\s*-{2,}|-{2,}\s*Forwarded Message\s*-{2,})", re.MULTILINE)


# EMAIL_START_BOUNDARY = re.compile(r"(?im)(?=^\s*From:|^\s*-{2,}\s*Original Message\s*-{2,}|^\s*On\s.+?\s+wrote:)", re.MULTILINE)
# EMAIL_START_BOUNDARY = re.compile(r"(?im)(?=^[\t >\u00A0]*From:|^[\t >\u00A0]*On\s+.+?\s+wrote:|^[\t >\u00A0]*[-]{2,}\s*Original Message)", re.MULTILINE)
HEADER_PATTERNS = {
    "from": re.compile(r"^From:\s*(.+)", re.IGNORECASE),
    "to": re.compile(r"^To:\s*(.+)", re.IGNORECASE),
    "cc": re.compile(r"^CC:\s*(.+)", re.IGNORECASE),
    "bcc": re.compile(r"^BCC:\s*(.+)", re.IGNORECASE),
    "subject": re.compile(r"^Subject:\s*(.+)", re.IGNORECASE),
    "date": re.compile(r"^Date:\s*(.+)", re.IGNORECASE),
    }

class EmailDuplicateFinder:
    """
    Extracts emails from an AnalyzeResult object and identifies any duplicates (e.g. in a stacked email thread).
    """
    def __init__(self, analysis_result: AnalyzeResult):
        """
        Initializes the finder with the AnalyzeResult object (output of the Document Intelligence parsing the input pdf) to search.
        """
        self.analysis_result = analysis_result


    def find_duplicate_emails(self, verbose = True) -> list:
        """
        Checks the document for one or more emails. If emails are present, scans for duplicate emails
        in stacked email threads. Returns a list of dicts containing the text and location of each duplicate email.
        Returns:
            List: A list of detailed duplicate email information. Each dict inside the list has the following fields:
                - 'id': unique identifier for the duplicate email
                - 'text': the text content of the duplicate email
                - 'category': the category of the redaction (e.g., "Duplicate Email")
                - 'reasoning': explanation for why it was flagged
                - 'context': surrounding context of the duplicate email
                - 'page_num': the page number where the duplicate email is located
                - 'rects': list of bounding rectangles for the duplicate email on the page
        """
        # Check doc for emails
        if self.__contains_email_message():
            if verbose: print("Document contains email messages. Scanning for duplicate emails...")
        else:
            if verbose: print("No email messages detected in document.")
            return
        
        # Parse email thread and extract emails
        try:
            emails = self.__parse_thread(self.analysis_result.content, verbose=verbose)
            duplicates = self.__get_duplicate_emails(emails)
            if verbose: print(f"Found {len(duplicates)} duplicate emails in the document.")
        except Exception as e:
            if verbose: print(f"Error extracting emails and scanning for duplicates: {e}")
            return
        
        # Match duplicates to their locations in the document
        duplicate_indices = set()
        for dup in duplicates:
            duplicate_indices.add(dup['target_email_index'])
        duplicate_emails = []
        for i, email in enumerate(emails):
            if i in duplicate_indices:
                duplicate_emails.append(email)
        
        # print(f"Found {len(duplicate_emails)} duplicate email instances to detail.")
        
        return self.__create_detailed_duplicate_emails(duplicate_emails)


    # -------- CHECK IF DOCUMENT CONTAINS EMAIL MESSAGE - main method (__contains_email_message) and helpers --------
    def __contains_email_message(self, *, debug: bool = False) -> bool:
        """
        Checks if the document likely contains at least one full email (header block + body), without requiring thread markers.
        Parameters:
            debug (bool): If True, prints debug information about detection steps.
        Returns:
            bool: True if an email message is detected, False otherwise.
        """
        text = self.__safe_get_text()
        if not text or not text.strip():
            return False

        # Preserve leading whitespace for correct continuation handling
        lines = text.splitlines()

        i, n = 0, len(lines)
        while i < n:
            line = lines[i]

            # Thread markers (optional)
            hdr_end, headers = self.__consume_header_block(lines, i + 1)
            if self.__is_email_header(headers) and self.__is_email_body_present(lines, hdr_end):
                if debug:
                    print(f"Email detected after 'Original Message' at {i}-{hdr_end} with {list(headers.keys())}")
                return True
            i = hdr_end
            continue

            if ON_WROTE_RE.match(line.strip()):
                neighborhood = " ".join(lines[max(0, i - 1): min(n, i + 2)])
                if EMAIL_RE.search(neighborhood) and self.__is_email_body_present(lines, i + 1, window=80, min_chars=40):
                    if debug:
                        print(f"Email detected via 'On … wrote:' at {i}")
                    return True
                i += 1
                continue

            # Standalone header block anywhere
            hname, _ = self.__match_header(line.lstrip())
            if hname:
                hdr_end, headers = self.__consume_header_block(lines, i)
                if debug:
                    # print parsed headers for diagnostics
                    print(f"Header block at {i}-{hdr_end}: {headers}")
                if self.__is_email_header(headers) and self.__is_email_body_present(lines, hdr_end, window=150, min_chars=40):
                    if debug:
                        print(f"Email detected via header block at {i}-{hdr_end} with {list(headers.keys())}")
                    return True
                # Skip past what we consumed to keep scanning
                i = max(i + 1, hdr_end)
                continue

            i += 1

        return False


    def __safe_get_text(self) -> str:
        """
        Get text content - prefer paragraph content, skipping page header/footer if present. Fall back to analyze_result.content.
        Returns:
            str: The extracted text content.
        """
        text: Optional[str] = getattr(self.analysis_result, "content", None)
        paragraphs = getattr(self.analysis_result, "paragraphs", None)
        if paragraphs:
            parts = []
            for p in paragraphs:
                p_text = getattr(p, "content", None)
                p_role = getattr(p, "role", None)
                if not p_text:
                    continue
                if p_role and str(p_role).lower() in {"pageheader", "pagefooter", "footnote"}:
                    continue
                parts.append(p_text)
            if parts:
                return "\n".join(parts)
        return text or ""


    def __is_email_header(self, headers: Dict[str, str]) -> bool:
        """
        Checks if the headers match format of email headers. Require a plausible combination. Typical emails have at least 2 of:
        From, To, Subject, Date/Sent (optionally CC/BCC/Reply-To). Technical headers can help, but are not required.
        Parameters:
            headers (Dict[str, str]): The headers to evaluate.
        Returns:
            bool: True if headers look like email headers, False otherwise.
        """
        key = {"from", "to", "subject", "date", "cc", "bcc", "reply_to"}
        tech = {"message_id", "mime_version", "content_type"}
        count_key = len(key & headers.keys())
        count_tech = len(tech & headers.keys())
        # Loosen constraints slightly to accommodate exports with minimal headers
        return (count_key >= 2) or (count_key >= 1 and count_tech >= 1)


    def __is_email_body_present(self, lines: List[str], start: int, *, window: int = 120, min_chars: int = 40) -> bool:
        """
        Checks if an email body is present. Look for body-like content in the next 'window' lines after 'start'.
        Uses *cumulative* characters to handle short, wrapped lines from PDFs. Stops when it hits another obvious header block/separator.
        Parameters:
            lines (List[str]): The list of lines to check.
            start (int): The starting index in lines to begin checking.
            window (int): The maximum number of lines to check.
            min_chars (int): The minimum number of alphanumeric characters required to consider body present.
        Returns:
            bool: True if email body-like content is detected, False otherwise.
        """
        total = 0
        n = len(lines)
        for i in range(start, min(n, start + window)):
            ln = lines[i]
            if not ln.strip():
                continue
            if ORIGINAL_MSG_RE.match(ln.strip()):
                break  # likely the start of another section
            # If next line looks like a header, stop unless we've already found enough body
            hname, _ = self.__match_header(ln.lstrip())
            if hname:
                return total >= min_chars
            # Count alnum content
            if re.search(r"[A-Za-z0-9]", ln):
                total += len(ln.strip())
                if total >= min_chars:
                    return True
        return total >= min_chars


    # -------- EMAIL PARSING (extract each email from text) - main method (__parse_thread) and helpers --------

    def __parse_thread(self, full_text: str, verbose = False) -> List[Dict[str, Any]]:
        """
        Parses a full email thread text into individual email blocks.
        Parameters:
            full_text (str): The complete text of the email thread.
        Returns:
            List[Dict[str, Any]]: A list of email blocks with their content and spans.
        """
        blocks = self.__split_thread_into_email_blocks(full_text)
        emails = []
        for start, block in blocks:
            emails.append(self.__parse_email_block(block, start))
        if verbose:
            print(f"Found {len(emails)} email blocks in thread.")
        return emails

    
    def __split_thread_into_email_blocks(self, full_text: str):
        """
        Splits stacked email threads into blocks by detecting header starts or quoted markers.
        Handles cases where middle emails only have 'On ... wrote:' markers.
        Parameters:
            full_text (str): The complete text of the email thread.
        Returns:
            List[Tuple[int, str]]: A list of email blocks with their starting indices and text.
        """
        text = full_text.strip()

        # Marker that delineates start of an email block. Here, email blocks can start with 'From:' or 'On ... wrote:'
        # marker = re.compile(r"(?i)(?=From:\s*[A-Z]|On\s+.+?\s+wrote:)", re.MULTILINE)

        indices = [m.start() for m in EMAIL_START_BOUNDARY.finditer(text)]
        if not indices:
            return [(0, text)]

        blocks = []
        for idx, start in enumerate(indices):
            end = indices[idx + 1] if idx + 1 < len(indices) else len(text)
            blocks.append((start, text[start:end].strip()))
        return blocks


    def __parse_email_block(self, block_text: str, global_offset: int) -> Dict[str, Any]:
        """
        Parse a single email block into headers and body.
        Parameters:
            block_text (str): The text of the email block.
            global_offset (int): The offset of the block in the full thread text.
        Returns:
            Dict: A dictionary with 'headers', 'body', 'content', and 'span'
        """
        lines = block_text.splitlines()
        headers = {}
        body_lines = []
        in_headers = True

        # header_patterns = {
        #     "from": re.compile(r"^From:\s*(.+)", re.IGNORECASE),
        #     "to": re.compile(r"^To:\s*(.+)", re.IGNORECASE),
        #     "cc": re.compile(r"^CC:\s*(.+)", re.IGNORECASE),
        #     "bcc": re.compile(r"^BCC:\s*(.+)", re.IGNORECASE),
        #     "subject": re.compile(r"^Subject:\s*(.+)", re.IGNORECASE),
        #     "date": re.compile(r"^Date:\s*(.+)", re.IGNORECASE),
        # }

        current_offset = global_offset
        for line in lines:
            line_len = len(line) + 1  # +1 for newline
            if in_headers:
                matched = False
                for key, pattern in HEADER_PATTERNS.items():
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

        headers = self.__get_headers(block_text)
        body_text = "\n".join(body_lines).strip()
        return {
            "headers": headers,
            "body": body_text,
            "content": block_text,
            "span": {"offset": global_offset, "length": len(block_text)}
        }


    def __get_headers(self, text_block: str) -> List[Dict[str, str]]:
        """
        Helper that extracts all header-like blocks from the document. Returns a list of dicts of headers.
        Parameters:
            text_block (str): The text block to extract headers from.
        Returns:
            List[Dict[str, str]]: A list of header dictionaries.
        """
        headers_list = []
        lines = text_block.splitlines()

        i, n = 0, len(lines)
        while i < n:
            line = lines[i]

            hdr_end, headers = self.__consume_header_block(lines, i)
            if headers:
                headers_list.append(headers)
                i = hdr_end
            else:
                i += 1

        return headers_list


    def __consume_header_block(self, lines: List[str], start: int, max_span: int = 60) -> Tuple[int, Dict[str, str]]:
        """
        Consume a contiguous block of headers, splitting lines that contain multiple headers.
        Parameters:
            lines (List[str]): The list of lines to process.
            start (int): The starting index in lines to begin processing.
            max_span (int): The maximum number of lines to process in this block.
        Returns:
            Tuple[int, Dict[str, str]]: The index of the line after the header block and a dictionary of headers.
        """
        headers: Dict[str, str] = {}
        i = start
        end = min(len(lines), start + max_span)
        blanks_tolerated = 0

        while i < end:
            raw = lines[i]
            line = raw if isinstance(raw, str) else self.__normalize_text(raw)

            if not line.strip():
                blanks_tolerated += 1
                if blanks_tolerated <= 1:
                    i += 1
                    continue
                break

            # Split line into possible multiple headers
            parts = re.split(r"(?=\b(?:From|To|Subject|Date|Sent|Cc|Bcc|Reply-To)\b\s*:)", line)
            matched_any = False
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                hname, hval = self.__match_header(part)
                if hname:
                    headers[hname] = (headers.get(hname, "") + " " + hval).strip()
                    matched_any = True

            if not matched_any:
                # Non-header line ends the block
                break

            i += 1

        return i, headers


    def __match_header(self, line: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to match an email-style header in the *original line*. Returns (canonical_header_name, value) or (None, None).
        Parameters:
            line (str): The line to match.
        Returns:
            Tuple[Optional[str], Optional[str]]: The canonical header name and its value, or (None, None) if no match.
        """
        m = HEADER_LINE_RE.match(line)
        if not m:
            return None, None
        raw_name, value = m.group(1), m.group(2)
        canon = self.__canon_header_name(raw_name)
        if not canon:
            return None, None
        return canon, value.strip()


    def __canon_header_name(self, raw: str) -> Optional[str]:
        """
        Maps various header name synonyms to canonical names.
        Parameters:
            raw (str): The raw header name.
        Returns:
            Optional[str]: The canonical header name, or None if unknown.
        """
        name = raw.strip().lower()
        name = re.sub(r"\s+", " ", name)      # collapse spaces
        name = name.replace("–", "-").replace("—", "-")  # normalize dashes

        synonyms = {
            "from": "from",
            "to": "to",
            "cc": "cc",
            "bcc": "bcc",
            "subject": "subject",
            "date": "date",
            "sent": "date",               # Outlook exports
            "reply-to": "reply_to",
            "reply to": "reply_to",
            "message-id": "message_id",
            "mime-version": "mime_version",
            "content-type": "content_type",
            "attachments": "attachments",
            "attachment": "attachments",
        }
        # also normalize awkward spacing around hyphens (e.g., "reply - to")
        name = name.replace(" - ", "-")
        return synonyms.get(name, None)  # return None for unknown headers

    def __normalize_text(self, s: str) -> str:
        """
        Normalizes a string for comparison (strings generated from PDFs can be weirdly/inconsistently formatted): lowercases and collapses whitespace.
        Parameters:
            s (str): The string to normalize.
        Returns:
            str: The normalized string.
        """
        # Lowercase + collapse whitespace; keep it simple and stable
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    # -------- FIND DUPLICATE EMAILS - main method (__get_duplicate_emails) and helpers --------

    def __get_duplicate_emails(
        self,
        emails: List[Dict[str, Any]],
        text_key: str = "body",
        min_chars: int = 30,
        fuzzy: bool = True,
        fuzzy_coverage: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Identify duplicates where later emails include the content of earlier emails (quoted replies).
        
        Parameters:
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

        duplicates = []

        # Precompute normalized texts
        norm_texts = [self.__normalize_text(e.get(text_key, "")) for e in emails]
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
                    span = self.__index_in_original(tgt_norm, tgt_orig, src_norm, src_orig)
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
                    match = self.__fuzzy_matching(emails[j].get(text_key, ""), j, i, src_norm, tgt_norm, tgt_orig, fuzzy_coverage)
                    if match:
                        duplicates.append(match)

        return duplicates


    
    def __index_in_original(self, norm_target: str, orig_target: str, norm_source: str, orig_source: str) -> Optional[tuple]:
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


    def __fuzzy_matching(self, email_content: str, src_index: int, tgt_index: int, src_norm: str, tgt_norm: str, tgt_orig: str, fuzzy_coverage: float) -> Dict[str, Any]:
        """
        Helper that performs fuzzy matching between source and target strings.
        Parameters:
            email_content (str): The content of the source email being checked.
            src_index (int): The index of the source email.
            tgt_index (int): The index of the target email.
            src_norm (str): The normalised source string.
            tgt_norm (str): The normalised target string.
            tgt_orig (str): The original target string.
            fuzzy_coverage (float): The required coverage threshold.
        Returns:
            Dict[str, Any]: A dictionary with match details if a match is found, otherwise None.
        """
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
            span = self.__index_in_original(tgt_norm, tgt_orig, window_norm, tgt_orig[a_start:a_end])
            return {
                "source_email_index": src_index, 
                "target_email_index": tgt_index, 
                "type": "fuzzy", 
                "coverage": round(coverage, 3), 
                "content": emails[j].get(text_key, ""), 
                "target_span": span if span else (0, 0)
                }
        else:
            return None


    # -------- GET BOUNDING BOX/POSITION INFO FOR EACH EMAIL DUPLICATE for pdf overlay (__create_detailed_duplicate_emails) - main method and helpers --------
    
    def __create_detailed_duplicate_emails(self, duplicate_emails: list) -> list:
        """
        Appends detailed information to each duplicate email, including page numbers and bounding boxes (for drawing boxes on pdf overlay).
        Parameters:
            duplicate_emails (list): List of duplicate email dicts with basic info.
        Returns:
            list: List of detailed duplicate email dicts with page numbers and bounding boxes.
        """
        detailed_duplicates = []
        scaling_factor = 72.0

        for idx, email in enumerate(duplicate_emails):
            span = email.get("body_span") or email.get("span")
            if not span:
                continue

            # Convert start and end offsets to page numbers and bounding boxes
            start = span["offset"]
            end = start + span["length"]

            page_boxes = self.__per_page_boxes_for_span(start, end)

            # Convert bounding boxes to fitz.Rect objects (for pdf overlay) and apply scaling
            # for page_num, box in page_boxes.items():
            #     rect = None
            #     if box:
            #         rect = fitz.Rect(
            #             box["x1"] * scaling_factor,
            #             box["y1"] * scaling_factor,
            #             box["x2"] * scaling_factor,
            #             box["y2"] * scaling_factor
            #         )

            #     detailed_duplicates.append({
            #         "id": f"dupemail_{idx}_{page_num}",
            #         "text": email.get("body") or email.get("content", ""),
            #         "category": "Duplicate Email",
            #         "reasoning": "Detected as duplicate email content in thread.",
            #         "context": email.get("content", ""),
            #         "page_num": page_num - 1,
            #         "rects": [rect] if rect else []
            #     })
            
            # ---- Create a single output object per email ----
            merged_entry = {
                "id": f"dupemail_{idx}",
                "text": email.get("body") or email.get("content", ""),
                "category": "Duplicate Email",
                "reasoning": "Detected as duplicate email content in thread.",
                "context": email.get("content", ""),
                "rects": []      # list of {page_num, rect}
            }

            # ---- Attach one rect per page to this single entry ----
            for page_num, box in page_boxes.items():
                if box:
                    rect = fitz.Rect(
                        box["x1"] * scaling_factor,
                        box["y1"] * scaling_factor,
                        box["x2"] * scaling_factor,
                        box["y2"] * scaling_factor
                    )
                    merged_entry["rects"].append({
                        "page_num": page_num - 1,
                        "rect": rect
                    })

            detailed_duplicates.append(merged_entry)

        return detailed_duplicates


    def __per_page_boxes_for_span(self, start_offset: int, end_offset: int) -> Dict[int, Dict[str, float]]:
        """
        Helper that returns merged rectangles per page that intersect [start_offset, end_offset).
        Output format:
        { pageNumber: {"x1":..., "y1":..., "x2":..., "y2":...}, ... }
        Coordinates are in the same units as Document Intelligence polygons.
        Parameters:
            start_offset (int): The start offset of the target text span.
            end_offset (int): The end offset of the target text span.
        Returns:
            Dict[int, Dict[str, float]]: A mapping of page numbers to their bounding boxes
        """
        by_page_points = {}  # page_num -> list[(x,y)]

        for page in self.analysis_result.pages:
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
        print("Calculating per-page bounding boxes for duplicate email spans...")
        for page_num, pts in by_page_points.items():
            xs = [x for x, y in pts]
            ys = [y for x, y in pts]
            per_page_boxes[page_num] = {
                "x1": min(xs), "y1": min(ys),  # top-left
                "x2": max(xs), "y2": max(ys),  # bottom-right
            }
            print(f"Page {page_num} box: {per_page_boxes[page_num]}")
        return per_page_boxes
