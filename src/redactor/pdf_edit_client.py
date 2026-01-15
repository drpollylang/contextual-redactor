# pdf_edit_client.py
import base64
import json
import requests
import fitz
from typing import List, Dict, Any

def canvas_obj_to_page_rect(obj: Dict[str, Any], zoom: float) -> Dict[str, float]:
    """Fabric.js rect -> page coords."""
    left = obj.get("left", 0.0)
    top = obj.get("top", 0.0)
    width = obj.get("width", 0.0) * obj.get("scaleX", 1.0)
    height = obj.get("height", 0.0) * obj.get("scaleY", 1.0)
    return {
        "x0": left / zoom,
        "y0": top / zoom,
        "x1": (left + width) / zoom,
        "y1": (top + height) / zoom
    }

def rect_to_payload(rect: fitz.Rect) -> Dict[str, float]:
    return {"x0": float(rect.x0), "y0": float(rect.y0), "x1": float(rect.x1), "y1": float(rect.y1)}

def call_pdf_edit_api(pdf_bytes: bytes, operations: List[Dict], api_url: str, font_mode="auto") -> bytes:
    payload = {
        "pdf_b64": base64.b64encode(pdf_bytes).decode("utf-8"),
        "font_mode": font_mode,
        "operations": operations
    }
    r = requests.post(f"{api_url.rstrip('/')}/v1/edit/remove-and-reflow",
                      headers={"Content-Type": "application/json"},
                      data=json.dumps(payload),
                      timeout=120)
    r.raise_for_status()
    data = r.json()
    return base64.b64decode(data["pdf_b64"])
