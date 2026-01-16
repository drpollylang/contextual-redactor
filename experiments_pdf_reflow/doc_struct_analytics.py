import fitz

doc = fitz.open("FakeStackedEmailThread.pdf")

for i in [0, 1]:   # page 1 is index 0, page 2 is index 1
    page = doc[i]
    print(f"\n=== PAGE {i+1} INFO ===")
    print("mediabox:", page.mediabox)
    print("cropbox:", page.cropbox)
    print("rotation:", page.rotation)

    print("\n-- Raw lines --")
    raw = page.get_text("rawdict")
    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            spans = ln.get("spans", [])
            chars = ln.get("chars", [])
            if spans:
                text = "".join(sp.get("text","") for sp in spans if "text" in sp)
            elif chars:
                text = "".join(ch.get("c","") for ch in chars)
            else:
                text = ""
            if text.strip():
                print(f"  line: {repr(text)}  | bbox:", ln["bbox"])

    print("\n-- Bottom-most Y (actual content) --")
    bottoms = []
    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                if "bbox" in sp:
                    bottoms.append(sp["bbox"][3])
            for ch in ln.get("chars", []):
                bottoms.append(ch["bbox"][3])
    print("max content Y:", max(bottoms) if bottoms else None)

    print("-- Top-most Y --")
    tops = []
    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            for sp in ln.get("spans", []):
                if "bbox" in sp:
                    tops.append(sp["bbox"][1])
            for ch in ln.get("chars", []):
                tops.append(ch["bbox"][1])
    print("min content Y:", min(tops) if tops else None)
