import os
import sys
import csv
import re
import pdfplumber

from statistics import mean


def is_number(text):
    return bool(re.match(r"^\d+(\.\d+)?$", text.strip()))


def is_all_caps(text):
    return text.isupper() and len(text) > 1


def is_valid_text(text):
    if not text.strip():
        return False
    non_ascii_count = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
    if non_ascii_count / max(len(text), 1) > 0.5:
        return False
    return True


def get_font_ranking(doc):
    """Rank font sizes (largest -> smallest)."""
    font_sizes = []
    for page in doc.pages:
        for char in page.chars:
            try:
                size = float(char.get("size", 0))
                if size > 0:
                    font_sizes.append(round(size, 1))
            except (ValueError, TypeError):
                continue
    unique_sizes = sorted(set(font_sizes), reverse=True)
    return {size: rank + 1 for rank, size in enumerate(unique_sizes)}


def merge_chars_to_elements(page, font_rank):
    """
    Groups pdfplumber characters into text elements based on font, size, and proximity.
    This version merges across line breaks if the style/size is the same.
    """
    elements = []
    chars = page.chars
    if not chars:
        return elements

    # Sort chars by vertical position then horizontal
    chars.sort(key=lambda c: (round(c.get("top", 1)), c.get("x0", 0)))

    current_element = None
    last_bottom = None

    for c in chars:
        text = c.get("text", "")
        if not is_valid_text(text):
            continue

        font_name = c.get("fontname", "")
        try:
            font_size = round(float(c.get("size", 0)), 1)
        except (ValueError, TypeError):
            font_size = 0

        is_bold = "Bold" in font_name or "bold" in font_name
        is_italic = "Italic" in font_name or "Oblique" in font_name
        is_underline = False

        try:
            x0 = float(c.get("x0", 0))
            x1 = float(c.get("x1", 0))
            top = float(c.get("top", 0))
            bottom = float(c.get("bottom", 0))
        except (ValueError, TypeError):
            continue

        bbox = (x0, top, x1, bottom)

        char_elem = {
            "text": text,
            "font": font_name,
            "font_size": font_size,
            "norm_font_size": font_rank.get(font_size, len(font_rank)),
            "is_bold": is_bold,
            "is_italic": is_italic,
            "is_underline": is_underline,
            "bbox": bbox,
        }

        if current_element:
            same_font = abs(current_element["font_size"] - font_size) <= 0.5
            same_style = (
                current_element["is_bold"] == is_bold
                and current_element["is_italic"] == is_italic
            )

            vertical_gap = abs(bbox[1] - last_bottom) if last_bottom is not None else 0
            horizontal_gap = bbox[0] - current_element["bbox"][2]

            # Merge across new lines if font and style match and gap is small
            if same_font and same_style and vertical_gap < (font_size * 1.5):
                if horizontal_gap > 5:
                    current_element["text"] += " " + text
                else:
                    current_element["text"] += text

                current_element["bbox"] = (
                    min(current_element["bbox"][0], bbox[0]),
                    min(current_element["bbox"][1], bbox[1]),
                    max(current_element["bbox"][2], bbox[2]),
                    max(current_element["bbox"][3], bbox[3]),
                )
                last_bottom = bbox[3]
                continue
            else:
                elements.append(current_element)

        current_element = char_elem
        last_bottom = bbox[3]

    if current_element:
        elements.append(current_element)

    for e in elements:
        e["is_all_caps"] = is_all_caps(e["text"])

    return elements


def filter_and_transform_rows(rows):
    filtered = []
    for row in rows:
        text = row["text"]
        wc = row["word_count"]
        if is_number(text):
            continue
        if wc <= 2 and not row["is_all_caps"]:
            continue

        row["is_bold"] = 1 if row["is_bold"] else 0
        row["is_italic"] = 1 if row["is_italic"] else 0
        row["is_underline"] = 1 if row["is_underline"] else 0
        row["is_all_caps"] = 1 if row["is_all_caps"] else 0
        filtered.append(row)
    return filtered


def process_pdf(pdf_path, output_csv):
    with pdfplumber.open(pdf_path) as pdf:
        font_rank = get_font_ranking(pdf)
        rows = []

        for page_num, page in enumerate(pdf.pages, start=1):
            elements = merge_chars_to_elements(page, font_rank)
            pw, ph = page.width, page.height

            for e in elements:
                text = e["text"]
                x0, y0, x1, y1 = e["bbox"]

                rows.append({
                    "page_number": page_num,
                    "is_bold": e["is_bold"],
                    "is_italic": e["is_italic"],
                    "is_underline": e["is_underline"],
                    "is_all_caps": e["is_all_caps"],
                    "word_count": len(text.split()),
                    "norm_font_size": e["norm_font_size"],
                    "font": e["font"],
                    "font_size": e["font_size"],
                    "x0": round(x0, 2), "y0": round(y0, 2),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "space_above": round(y0, 2),
                    "space_below": round(ph - y1, 2),
                    "space_left": round(x0, 2),
                    "space_right": round(pw - x1, 2),
                    "text": text
                })

    final_rows = filter_and_transform_rows(rows)

    fieldnames = [
        "page_number", "is_bold", "is_italic", "is_underline", "is_all_caps",
        "word_count", "norm_font_size", "font", "font_size",
        "x0", "y0", "x1", "y1",
        "space_above", "space_below", "space_left", "space_right", "text"
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fn in os.listdir(input_folder):
        if fn.lower().endswith(".pdf"):
            inpath = os.path.join(input_folder, fn)
            outpath = os.path.join(output_folder, fn.replace(".pdf", ".csv"))
            print(f"Processing: {inpath}")
            process_pdf(inpath, outpath)
            print(f"Saved: {outpath}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)
    process_folder(sys.argv[1], sys.argv[2])
