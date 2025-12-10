import os
import sys
import csv
import re
from collections import Counter
from pypdf import PdfReader

# -----------------------------
# Heuristics & helpers
# -----------------------------

def is_number(text: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)?$", text.strip()))

def is_all_caps(text: str) -> bool:
    return text.isupper() and len(text) > 1

def is_valid_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    non_ascii_count = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
    if non_ascii_count / max(len(text), 1) > 0.5:
        return False
    return True

# -----------------------------
# Low-level extraction (tokens)
# -----------------------------

def parse_pdf_tokens(pdf_path):
    """
    Extract tokens (tiny text chunks) with font, size and position using pypdf visitor.
    Returns (tokens, reader). Each token: {page, text, x, y, font, size, bold, italic}
    """
    reader = PdfReader(pdf_path)
    tokens = []

    for page_idx, page in enumerate(reader.pages, start=1):
        def visitor(text, cm, tm, font_dict, font_size):
            if not text or not is_valid_text(text):
                return
            # Preserve spaces (carry layout info) but drop newlines/tabs.
            t = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            x, y = float(tm[4]), float(tm[5])

            base_font = font_dict.get("/BaseFont") if isinstance(font_dict, dict) else None
            font_name = str(base_font) if base_font else "Unknown"
            is_bold = ("Bold" in font_name) or ("bold" in font_name)
            is_italic = ("Italic" in font_name) or ("Oblique" in font_name)

            tokens.append({
                "page": page_idx,
                "text": t,
                "x": x,
                "y": y,
                "font": font_name,
                "size": float(font_size) if font_size is not None else 0.0,
                "bold": is_bold,
                "italic": is_italic,
            })

        page.extract_text(visitor_text=visitor)

    return tokens, reader

# -----------------------------
# Layout: tokens -> lines -> paragraphs
# -----------------------------

def tokens_to_lines(tokens):
    """
    Group tokens on the same baseline into reading-order lines.
    Adds a space only when the X-gap suggests a word boundary,
    so split words like R + ESEARCH become RESEARCH.
    """
    # Sort by page, then by y (top->bottom), then x (left->right)
    tokens_sorted = sorted(tokens, key=lambda t: (t["page"], -t["y"], t["x"]))

    lines = []
    current = None

    def flush_current():
        nonlocal current
        if current is None:
            return
        # Dominant style on the line
        font = Counter(seg["font"] for seg in current["segs"]).most_common(1)[0][0]
        size = Counter(round(seg["size"], 2) for seg in current["segs"]).most_common(1)[0][0]
        bold = Counter(seg["bold"] for seg in current["segs"]).most_common(1)[0][0]
        italic = Counter(seg["italic"] for seg in current["segs"]).most_common(1)[0][0]
        text = current["text"].strip()

        x0 = min(seg["x0"] for seg in current["segs"])
        x1 = max(seg["x1"] for seg in current["segs"])
        y_top = max(seg["y"] for seg in current["segs"])           # top (larger y in PDF coords)
        y_bottom = min(seg["y"] - seg["size"] for seg in current["segs"])

        lines.append({
            "page": current["page"],
            "text": text,
            "font": font,
            "size": float(size),
            "bold": bool(bold),
            "italic": bool(italic),
            "x0": float(x0), "y0": float(y_bottom),
            "x1": float(x1), "y1": float(y_top),
        })
        current = None

    for tok in tokens_sorted:
        # rough width estimate (good enough for gap heuristic)
        est_w = max(0.0, len(tok["text"]) * (tok["size"] * 0.5))
        x0, x1 = tok["x"], tok["x"] + est_w
        y = tok["y"]

        if current is None:
            current = {
                "page": tok["page"],
                "text": tok["text"],
                "y_ref": y,
                "size_ref": tok["size"],
                "x0": x0,
                "x1": x1,
                "segs": [{
                    "font": tok["font"], "size": tok["size"],
                    "bold": tok["bold"], "italic": tok["italic"],
                    "x0": x0, "x1": x1, "y": y,
                }],
            }
            continue

        # New page -> flush
        if tok["page"] != current["page"]:
            flush_current()
            current = {
                "page": tok["page"],
                "text": tok["text"],
                "y_ref": y, "size_ref": tok["size"],
                "x0": x0, "x1": x1,
                "segs": [{
                    "font": tok["font"], "size": tok["size"],
                    "bold": tok["bold"], "italic": tok["italic"],
                    "x0": x0, "x1": x1, "y": y,
                }],
            }
            continue

        # Same page: check baseline proximity (same line?)
        y_tol = max(2.0, 0.4 * max(current["size_ref"], tok["size"]))  # tune if needed
        same_line = abs(y - current["y_ref"]) <= y_tol

        if not same_line:
            flush_current()
            current = {
                "page": tok["page"],
                "text": tok["text"],
                "y_ref": y, "size_ref": tok["size"],
                "x0": x0, "x1": x1,
                "segs": [{
                    "font": tok["font"], "size": tok["size"],
                    "bold": tok["bold"], "italic": tok["italic"],
                    "x0": x0, "x1": x1, "y": y,
                }],
            }
            continue

        # Same line: add a space only if x-gap indicates a new word
        gap = max(0.0, x0 - current["x1"])     # negative => overlap
        no_space_thresh = 0.15 * tok["size"]   # tighten/loosen if needed
        space_needed = gap > no_space_thresh

        current["text"] += (" " if space_needed else "") + tok["text"]
        current["x1"] = max(current["x1"], x1)
        current["segs"].append({
            "font": tok["font"], "size": tok["size"],
            "bold": tok["bold"], "italic": tok["italic"],
            "x0": x0, "x1": x1, "y": y,
        })

    flush_current()
    return lines

def lines_to_paragraphs(lines):
    """
    Merge consecutive lines into paragraphs when font & style match and alignment is consistent.
    """
    paragraphs = []
    cur = None

    def flush():
        nonlocal cur
        if cur is None:
            return
        paragraphs.append({
            "page": cur["page"],
            "text": cur["text"].strip(),
            "font": cur["font"],
            "size": cur["size"],
            "bold": cur["bold"],
            "italic": cur["italic"],
            "x0": cur["x0"], "y0": cur["y0"],
            "x1": cur["x1"], "y1": cur["y1"],
        })
        cur = None

    for ln in lines:
        if cur is None:
            cur = {
                "page": ln["page"], "text": ln["text"],
                "font": ln["font"], "size": ln["size"],
                "bold": ln["bold"], "italic": ln["italic"],
                "x0": ln["x0"], "y0": ln["y0"], "x1": ln["x1"], "y1": ln["y1"],
                "last_left": ln["x0"], "last_y1": ln["y1"],
            }
            continue

        same_page = ln["page"] == cur["page"]
        same_style = (
            ln["font"] == cur["font"] and
            abs(ln["size"] - cur["size"]) <= 0.5 and
            ln["bold"] == cur["bold"] and
            ln["italic"] == cur["italic"]
        )
        if not same_page or not same_style:
            flush()
            cur = {
                "page": ln["page"], "text": ln["text"],
                "font": ln["font"], "size": ln["size"],
                "bold": ln["bold"], "italic": ln["italic"],
                "x0": ln["x0"], "y0": ln["y0"], "x1": ln["x1"], "y1": ln["y1"],
                "last_left": ln["x0"], "last_y1": ln["y1"],
            }
            continue

        # Paragraph merge thresholds (tune if you like)
        v_gap = ln["y1"] - cur["last_y1"]        # y increases upward in PDF coords
        v_tol = 1.5 * max(cur["size"], ln["size"])  # allow ~1.5 line heights
        left_shift = abs(ln["x0"] - cur["last_left"])
        left_tol = 1.2 * cur["size"]               # allow small indent drift

        if v_gap <= v_tol and left_shift <= left_tol:
            # Hyphenation join: strip hyphen at line end
            if cur["text"].rstrip().endswith("-"):
                cur["text"] = cur["text"].rstrip("- ") + ln["text"].lstrip()
            else:
                cur["text"] += " " + ln["text"].lstrip()

            cur["x0"] = min(cur["x0"], ln["x0"])
            cur["y0"] = min(cur["y0"], ln["y0"])
            cur["x1"] = max(cur["x1"], ln["x1"])
            cur["y1"] = max(cur["y1"], ln["y1"])
            cur["last_left"] = ln["x0"]
            cur["last_y1"] = ln["y1"]
        else:
            flush()
            cur = {
                "page": ln["page"], "text": ln["text"],
                "font": ln["font"], "size": ln["size"],
                "bold": ln["bold"], "italic": ln["italic"],
                "x0": ln["x0"], "y0": ln["y0"], "x1": ln["x1"], "y1": ln["y1"],
                "last_left": ln["x0"], "last_y1": ln["y1"],
            }

    flush()
    return paragraphs

# -----------------------------
# Font ranking (norm_font_size)
# -----------------------------

def build_font_rank(all_sizes):
    uniq = sorted(set(round(s, 2) for s in all_sizes), reverse=True)
    return {sz: i + 1 for i, sz in enumerate(uniq)}

# -----------------------------
# CSV building
# -----------------------------

def filter_and_transform_rows(elems, page_sizes):
    rows = []
    for el in elems:
        txt = el["text"]
        if not is_valid_text(txt):
            continue
        wc = len(txt.split())
        if is_number(txt):
            continue
        if wc <= 2 and not is_all_caps(txt):
            continue

        pg = el["page"]
        pw, ph = page_sizes[pg]
        x0, y0, x1, y1 = el["x0"], el["y0"], el["x1"], el["y1"]

        rows.append({
            "page_number": pg,
            "is_bold": 1 if el["bold"] else 0,
            "is_italic": 1 if el["italic"] else 0,
            "is_underline": 0,  # pypdf cannot detect underline cheaply
            "is_all_caps": 1 if is_all_caps(txt) else 0,
            "word_count": wc,
            "norm_font_size": el.get("norm_size", 0),
            "font": el["font"],
            "font_size": round(el["size"], 2),
            "x0": round(x0, 2), "y0": round(y0, 2),
            "x1": round(x1, 2), "y1": round(y1, 2),
            "space_above": round(y0, 2),
            "space_below": round(ph - y1, 2),
            "space_left": round(x0, 2),
            "space_right": round(pw - x1, 2),
            "text": txt,
        })
    return rows

# -----------------------------
# Main pipeline
# -----------------------------

def process_pdf(pdf_path, output_csv):
    tokens, reader = parse_pdf_tokens(pdf_path)

    # Per-page sizes
    page_sizes = {}
    for idx, page in enumerate(reader.pages, start=1):
        page_sizes[idx] = (float(page.mediabox.width), float(page.mediabox.height))

    # Build lines, then paragraphs
    lines = tokens_to_lines(tokens)
    paragraphs = lines_to_paragraphs(lines)

    # Rank font sizes across whole document
    all_sizes = [p["size"] for p in paragraphs]
    rank = build_font_rank(all_sizes)
    for p in paragraphs:
        p["norm_size"] = rank.get(round(p["size"], 2), len(rank) + 1)

    rows = filter_and_transform_rows(paragraphs, page_sizes)

    fieldnames = [
        "page_number", "is_bold", "is_italic", "is_underline", "is_all_caps",
        "word_count", "norm_font_size", "font", "font_size", "x0", "y0", "x1", "y1",
        "space_above", "space_below", "space_left", "space_right", "text"
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(".pdf"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".pdf", ".csv"))
            print(f"Processing: {input_path}")
            process_pdf(input_path, output_path)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_folder(input_folder, output_folder)
