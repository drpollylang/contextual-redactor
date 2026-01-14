
import json
import hashlib
import os
from typing import List, Dict, Any, Tuple
import numpy as np

# ---- Embeddings (Azure OpenAI) ----
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

# ---- Similarity Search (FAISS) ----
import faiss

# ---- PDF Visualization ----
import fitz  # PyMuPDF


# ---------------------------
# Configuration
# ---------------------------
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01-preview")
EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
SIMILARITY_THRESHOLD = 0.95      # cosine similarity threshold for near-duplicates
FAISS_TOP_K = 5                  # how many neighbors to retrieve per block
EMBED_BATCH_SIZE = 64            # batch size for embedding requests

# Visualization
BOX_COLOR = (1, 0, 0)            # red (RGB in 0..1)
BOX_FILL = (1, 0, 0)             # red fill
BOX_OPACITY = 0.25               # semi-transparent


# ---------------------------
# Helpers: text + geometry
# ---------------------------

# ===========================
# Helpers: text + geometry
# ===========================
EMAIL_HEADER_PATTERN = re.compile(
    r"(from:.*?$|sent:.*?$|to:.*?$|subject:.*?$|cc:.*?$|bcc:.*?$)",
    flags=re.IGNORECASE | re.MULTILINE
)

def normalize_text(text: Optional[str]) -> str:
    """Normalize text for duplicate comparison (email-friendly)."""
    if not text:
        return ""
    t = EMAIL_HEADER_PATTERN.sub("", text)      # strip common headers
    t = re.sub(r"^>+\s?", "", t, flags=re.MULTILINE)  # remove quote markers
    t = " ".join(t.lower().split())         # lower + collapse whitespace
    return t

    """Normalize text for duplicate comparison."""
    if text is None:
        return ""
    # Lowercase, collapse whitespace, strip email headers optionally
    t = " ".join(text.lower().split())
    return t

def polygon_to_bbox(polygon: List[Dict[str, float]]) -> Tuple[float, float, float, float]:
    """
    Convert polygon (list of points dicts or [x, y] pairs) to bounding box (x_min, y_min, x_max, y_max).
    Accepts either [{'x':..., 'y':...}, ...] OR [[x,y], ...].
    """
    xs, ys = [], []
    for p in polygon:
        if isinstance(p, dict):
            xs.append(p.get("x", 0.0))
            ys.append(p.get("y", 0.0))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))

def compute_size_from_bbox(bbox: Tuple[float, float, float, float]) -> Dict[str, float]:
    x0, y0, x1, y1 = bbox
    return {"width": max(0.0, x1 - x0), "height": max(0.0, y1 - y0)}

def make_relative_bbox(bbox: Tuple[float, float, float, float], page_w: float, page_h: float) -> Tuple[float, float, float, float]:
    """Convert absolute bbox to relative (0..1) coords given page width/height."""
    x0, y0, x1, y1 = bbox
    if page_w <= 0 or page_h <= 0:
        return (0, 0, 0, 0)
    return (x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h)



# ===========================
# Azure OpenAI client + embeddings
# ===========================
def get_azure_openai_client():
    if AzureOpenAI is None:
        raise ImportError("azure-openai not installed. `pip install azure-openai`")
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        # api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

def embed_paragraphs_batch(paragraphs: Sequence[Any]) -> np.ndarray:
    """
    NEW: Embed a batch of DocumentParagraph objects (already correctly sized).
    - paragraphs: a batch of `DocumentParagraph` as returned by the DI SDK.
      We read `p.content` and normalize before embedding.
    Returns a float32 array of shape [len(paragraphs), 1536] for ada-002.
    """
    client = get_azure_openai_client()

    # Prepare normalized texts
    inputs = [normalize_text(getattr(p, "content", None) or getattr(p, "text", None) or "") for p in paragraphs]
    if not any(inputs):
        return np.zeros((len(paragraphs), 1536), dtype=np.float32)

    # Simple exponential backoff
    for attempt in range(RETRY_MAX):
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_DEPLOYMENT,  # <-- deployment name, e.g., "text-embedding-ada-002"
                input=inputs
            )
            vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
            return np.vstack(vecs).astype(np.float32) if vecs else np.zeros((len(paragraphs), 1536), dtype=np.float32)
        except Exception as e:
            msg = str(e).lower()
            # Retry on common transient issues
            if any(token in msg for token in ["429", "rate", "temporarily", "timeout", "503"]):
                time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))
                continue
            raise


# ===========================
# FAISS helpers
# ===========================
def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def find_near_duplicates_faiss(vectors: np.ndarray, threshold: float, top_k: int) -> List[Tuple[int, int, float]]:
    """
    Return (i, j, similarity) for j > i with similarity >= threshold using cosine via inner product.
    """
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    sims, idxs = index.search(vectors, top_k)

    pairs = []
    n = vectors.shape[0]
    for i in range(n):
        for k in range(top_k):
            j = int(idxs[i, k])
            sim = float(sims[i, k])
            if j <= i:
                continue
            if sim >= threshold:
                pairs.append((i, j, sim))
    return pairs


# ===========================
# SDK path: paragraphs -> blocks
# ===========================
def extract_blocks_from_paragraphs(paragraphs: Sequence[Any], pages: Sequence[Any]) -> List[Dict[str, Any]]:
    """
    Convert DI SDK paragraphs + pages into 'blocks' with geometry and text.
    A paragraph can have multiple bounding regions (one block per region).
    """
    # Build page meta (1-based pageNumber in SDK)
    page_meta = {}
    for p in pages or []:
        pg_num = getattr(p, "page_number", None) or getattr(p, "pageNumber", None) or 0
        width = getattr(p, "width", None) or 1.0
        height = getattr(p, "height", None) or 1.0
        unit = getattr(p, "unit", None)
        page_meta[pg_num] = {"width": float(width), "height": float(height), "unit": unit}

    blocks = []
    for para_idx, para in enumerate(paragraphs):
        text = getattr(para, "content", None) or getattr(para, "text", None) or ""
        brs = getattr(para, "bounding_regions", None) or getattr(para, "boundingRegions", None) or []
        # If no regions, we still record a block without geometry for completeness
        if not brs:
            blocks.append({
                "paragraph_index": para_idx,
                "text": text,
                "page": None,
                "bbox_abs": (0, 0, 0, 0),
                "bbox_rel": (0, 0, 0, 0),
                "polygon": [],
                "page_size": None
            })
            continue

        for br in brs:
            pg_num = getattr(br, "page_number", None) or getattr(br, "pageNumber", None) or 0
            polygon = getattr(br, "polygon", None) or []
            bbox_abs = polygon_to_bbox(polygon)
            page_w = page_meta.get(pg_num, {}).get("width", 1.0)
            page_h = page_meta.get(pg_num, {}).get("height", 1.0)
            bbox_rel = make_relative_bbox(bbox_abs, page_w, page_h)
            blocks.append({
                "paragraph_index": para_idx,
                "text": text,
                "page": int(pg_num) - 1 if pg_num else None,  # convert to 0-based for PyMuPDF
                "bbox_abs": bbox_abs,
                "bbox_rel": bbox_rel,
                "polygon": polygon,
                "page_size": {"width": page_w, "height": page_h, "unit": page_meta.get(pg_num, {}).get("unit")}
            })
    return blocks


# ===========================
# Main: duplicate detection from SDK AnalyzeResult
# ===========================
def detect_duplicates_from_sdk_result(
    analyze_result: Any,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = FAISS_TOP_K,
    embed_batch_size: int = 64
) -> Dict[str, Any]:
    """
    Run exact + near-duplicate detection using SDK objects.
    - Uses `embed_paragraphs_batch` where the *caller-provided* paragraph batches are sized,
      but we show a simple batching loop here (you can replace with your own).
    """
    paragraphs: List[Any] = getattr(analyze_result, "paragraphs", None) or []
    pages: List[Any] = getattr(analyze_result, "pages", None) or []

    # Build blocks per bounding region
    blocks = extract_blocks_from_paragraphs(paragraphs, pages)

    # Exact duplicates by normalized text at the paragraph level
    norm_by_para = [normalize_text(getattr(p, "content", None) or getattr(p, "text", None) or "") for p in paragraphs]
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() if t else None for t in norm_by_para]

    para_seen: Dict[str, List[int]] = {}
    exact_dups = []
    for para_idx, h in enumerate(hashes):
        if not h or not norm_by_para[para_idx]:
            continue
        if h in para_seen:
            # map to first region of this paragraph (if any) for size/position
            blk = next((b for b in blocks if b["paragraph_index"] == para_idx), None)
            exact_dups.append({
                "paragraph_index": para_idx,
                "text": getattr(paragraphs[para_idx], "content", None) or "",
                "page": blk["page"] if blk else None,
                "bbox_abs": blk["bbox_abs"] if blk else (0, 0, 0, 0),
                "bbox_rel": blk["bbox_rel"] if blk else (0, 0, 0, 0),
                "size": compute_size_from_bbox(blk["bbox_abs"]) if blk else {"width": 0, "height": 0},
                "hash": h,
                "matches_with": para_seen[h]  # prior paragraph index(es)
            })
        else:
            para_seen[h] = [para_idx]

    # ---- Near-duplicates via embeddings + FAISS ----
    # Create embeddings for all paragraphs using caller-controlled batches.
    # Here, we implement a simple batching loop that calls embed_paragraphs_batch()
    # with batches of DocumentParagraph objects. If you already control batching upstream,
    # you can replace this loop with your own and just stack the vectors.
    all_vecs = []
    para_indices = []  # to keep mapping of vector row -> paragraph index
    for i in range(0, len(paragraphs), embed_batch_size):
        batch = paragraphs[i:i + embed_batch_size]
        vecs = embed_paragraphs_batch(batch)  # <--- uses DocumentParagraph batch directly
        all_vecs.append(vecs)
        para_indices.extend(range(i, min(i + embed_batch_size, len(paragraphs))))

    if all_vecs:
        vectors = np.vstack(all_vecs).astype(np.float32)   # shape [n_paragraphs, 1536]
    else:
        vectors = np.zeros((0, 1536), dtype=np.float32)

    if vectors.shape[0] == 0:
        near_dups = []
    else:
        pairs = find_near_duplicates_faiss(vectors, similarity_threshold, top_k)

        # Build output (choose the first block per paragraph for geometry; include all is optional)
        near_dups = []
        for li, lj, sim in pairs:
            i_para = para_indices[li]
            j_para = para_indices[lj]

            blk_i = next((b for b in blocks if b["paragraph_index"] == i_para), None)
            blk_j = next((b for b in blocks if b["paragraph_index"] == j_para), None)

            near_dups.append({
                "pair": [i_para, j_para],
                "similarity": round(float(sim), 4),
                "a": {
                    "text": getattr(paragraphs[i_para], "content", None) or "",
                    "page": blk_i["page"] if blk_i else None,
                    "bbox_abs": blk_i["bbox_abs"] if blk_i else (0, 0, 0, 0),
                    "bbox_rel": blk_i["bbox_rel"] if blk_i else (0, 0, 0, 0),
                    "size": compute_size_from_bbox(blk_i["bbox_abs"]) if blk_i else {"width": 0, "height": 0},
                },
                "b": {
                    "text": getattr(paragraphs[j_para], "content", None) or "",
                    "page": blk_j["page"] if blk_j else None,
                    "bbox_abs": blk_j["bbox_abs"] if blk_j else (0, 0, 0, 0),
                    "bbox_rel": blk_j["bbox_rel"] if blk_j else (0, 0, 0, 0),
                    "size": compute_size_from_bbox(blk_j["bbox_abs"]) if blk_j else {"width": 0, "height": 0},
                }
            })

    return {"exact": exact_dups, "near": near_dups, "paragraphs_count": len(paragraphs), "blocks_count": len(blocks)}

# ---------------------------
# Visualization on the PDF
# ---------------------------
def draw_boxes_on_pdf(input_pdf: str, output_pdf: str, duplicates: Dict[str, Any]):
    """
    Draw semi-transparent rectangles over duplicate blocks on the PDF.
    Uses bbox_rel (relative coords) to avoid unit mismatches between DI and PDF.
    """
    doc = fitz.open(input_pdf)

    def draw_block(page_idx: int, bbox_rel: Tuple[float, float, float, float], label: str):
        if page_idx < 0 or page_idx >= len(doc):
            return
        page = doc[page_idx]
        W = page.rect.width
        H = page.rect.height
        rx0, ry0, rx1, ry1 = bbox_rel
        rect = fitz.Rect(rx0 * W, ry0 * H, rx1 * W, ry1 * H)

        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=BOX_COLOR, fill=BOX_FILL, opacity=BOX_OPACITY)
        shape.commit()

        # Optional: add a small label near the box
        page.insert_text(
            rect.tl,  # top-left
            label,
            fontsize=8,
            color=(0, 0, 0),
            overlay=True
        )

    # Draw exact duplicates
    for i, dup in enumerate(duplicates.get("exact", []), start=1):
        page_idx = dup["page"]
        draw_block(page_idx, dup["bbox_rel"], f"EXACT #{i}")

    # Draw near duplicates (mark both sides of each pair)
    for i, pair in enumerate(duplicates.get("near", []), start=1):
        a = pair["a"]
        b = pair["b"]
        draw_block(a["page"], a["bbox_rel"], f"NEAR #{i} (A) sim={pair['similarity']}")
        draw_block(b["page"], b["bbox_rel"], f"NEAR #{i} (B) sim={pair['similarity']}")

    doc.save(output_pdf)
    doc.close()


# ---------------------------
# CLI example usage
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect and visualize duplicate text blocks in a PDF using Azure Document Intelligence + FAISS.")
    parser.add_argument("--di-json", required=True, help="Path to Azure Document Intelligence JSON output file.")
    parser.add_argument("--pdf", required=True, help="Original PDF file path.")
    parser.add_argument("--out-pdf", default="duplicates_highlighted.pdf", help="Output PDF with highlighted duplicates.")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Cosine similarity threshold.")
    parser.add_argument("--topk", type=int, default=FAISS_TOP_K, help="FAISS top-k neighbors per block.")
    args = parser.parse_args()

    with open(args.di_json, "r", encoding="utf-8") as f:
        di_json = json.load(f)

    results = detect_duplicates(di_json, similarity_threshold=args.threshold, top_k=args.topk)

    # Print a concise summary to console
    print(json.dumps({
        "blocks_count": results["blocks_count"],
        "exact_count": len(results["exact"]),
        "near_count": len(results["near"])
    }, indent=2))

    # Save detailed results (optional)
    with open("duplicates_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Visualize on PDF
    draw_boxes_on_pdf(args.pdf, args.out_pdf, results)
    print(f"Saved highlighted PDF to: {args.out_pdf}")
