import os
import sys
import csv
import fitz  # PyMuPDF
import re

def is_number(text):
    """Checks if a string is a number (integer or float)."""
    return bool(re.match(r"^\d+(\.\d+)?$", text.strip()))

def is_all_caps(text):
    """Checks if a string is all uppercase."""
    return text.isupper() and len(text) > 1

def get_font_ranking(doc):
    """
    Ranks font sizes found in the document from largest to smallest.
    Returns a dictionary mapping font size to its rank (1 for largest).
    """
    font_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    font_sizes.append(span["size"])
    unique_sizes = sorted(list(set(font_sizes)), reverse=True)
    return {size: rank + 1 for rank, size in enumerate(unique_sizes)}

def is_valid_text(text):
    """Checks if the text contains a significant amount of valid, printable characters."""
    if not text.strip():
        return False
    # Count non-ASCII printable characters
    non_ascii_count = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
    if non_ascii_count / max(len(text), 1) > 0.5:
        return False
    return True

def merge_spans_to_elements(page, font_rank):
    """
    Merges text spans into larger, more meaningful text elements based on
    font, style, and vertical proximity.
    """
    elements = []
    blocks = page.get_text("dict")["blocks"]

    current_element = None
    last_y1 = None

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            line_text = []
            font_sizes = []
            font_names = []
            is_bold_list = []
            is_italic_list = []
            is_underline_list = []
            bboxes = []

            for span in line["spans"]:
                text = span["text"].strip()
                if not is_valid_text(text):
                    continue

                font_name = span["font"]
                font_size = span["size"]

                is_bold = "Bold" in font_name or "bold" in font_name
                is_italic = "Italic" in font_name or "Oblique" in font_name
                is_underlined = bool(span.get("flags", 0) & 4)

                line_text.append(text)
                font_sizes.append(font_size)
                font_names.append(font_name)
                is_bold_list.append(is_bold)
                is_italic_list.append(is_italic)
                is_underline_list.append(is_underlined)
                bboxes.append(span["bbox"])

            if not line_text:
                continue

            merged_text = ' '.join(line_text)
            avg_font_size = sum(font_sizes) / len(font_sizes)
            font_name = font_names[0]
            is_bold = any(is_bold_list)
            is_italic = any(is_italic_list)
            is_underline = any(is_underline_list)
            x0 = min(b[0] for b in bboxes)
            y0 = min(b[1] for b in bboxes)
            x1 = max(b[2] for b in bboxes)
            y1 = max(b[3] for b in bboxes)

            element = {
                "text": merged_text,
                "font": font_name,
                "font_size": avg_font_size,
                "norm_font_size": font_rank.get(avg_font_size, len(font_rank)),
                "is_bold": is_bold,
                "is_italic": is_italic,
                "is_underline": is_underline,
                "bbox": (x0, y0, x1, y1),
                "is_all_caps": is_all_caps(merged_text),
            }

            # Merge with previous element if font/style is same and lines are close vertically
            if (
                current_element is not None
                and abs(current_element["font_size"] - element["font_size"]) <= 0.5
                and current_element["is_bold"] == element["is_bold"]
                and abs(element["bbox"][1] - last_y1) < 10
            ):
                current_element["text"] += ' ' + element["text"]
                current_element["bbox"] = (
                    min(current_element["bbox"][0], element["bbox"][0]),
                    current_element["bbox"][1],
                    max(current_element["bbox"][2], element["bbox"][2]),
                    element["bbox"][3],
                )
                last_y1 = element["bbox"][3]
            else:
                if current_element:
                    elements.append(current_element)
                current_element = element
                last_y1 = element["bbox"][3]

    if current_element:
        elements.append(current_element)

    return elements

def filter_and_transform_rows(rows):
    """
    Filters rows that are likely tables or equations and transforms
    boolean values to integers.
    """
    filtered_rows = []
    for row in rows:
        # 1. Heuristics for filtering tables and equations
        # A row is likely a table or equation if:
        # a) It contains only a number.
        # b) It has a very low word count (e.g., <= 2) and is not a heading (i.e., not all caps).
        word_count = row["word_count"]
        text = row["text"]
        is_heading = row["is_all_caps"]

        if is_number(text):
            # Exclude rows that are just numbers. This often indicates a table cell.
            continue
        if word_count <= 2 and not is_heading:
            # Exclude short, non-heading phrases, which could be table cells or equation numbers.
            continue
            
        # Images are not extracted by PyMuPDF's text extraction methods,
        # so no filtering is needed for them.

        # 2. Transform boolean values to 0s and 1s
        row["is_bold"] = 1 if row["is_bold"] else 0
        row["is_italic"] = 1 if row["is_italic"] else 0
        row["is_underline"] = 1 if row["is_underline"] else 0
        row["is_all_caps"] = 1 if row["is_all_caps"] else 0

        filtered_rows.append(row)
    return filtered_rows


def process_pdf(pdf_path, output_csv):
    """
    Processes a single PDF file, extracts text with metadata, and saves
    the filtered and transformed data to a CSV file.
    """
    doc = fitz.open(pdf_path)
    font_rank = get_font_ranking(doc)
    rows = []

    for page_num, page in enumerate(doc, start=1):
        page_height = page.rect.height
        page_width = page.rect.width

        elements = merge_spans_to_elements(page, font_rank)

        for elem in elements:
            text = elem["text"]
            x0, y0, x1, y1 = elem["bbox"]

            # Spacing and normalization
            space_above = y0
            space_below = page_height - y1
            space_left = x0
            space_right = page_width - x1

            rows.append({
                "page_number": page_num,
                "is_bold": elem["is_bold"],
                "is_italic": elem["is_italic"],
                "is_underline": elem["is_underline"],
                "is_all_caps": elem["is_all_caps"],
                "word_count": len(text.split()),
                "norm_font_size": elem["norm_font_size"],
                "font": elem["font"],
                "font_size": elem["font_size"],
                "x0": round(x0, 2), "y0": round(y0, 2),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "space_above": round(space_above, 2),
                "space_below": round(space_below, 2),
                "space_left": round(space_left, 2),
                "space_right": round(space_right, 2),
                "text": text,
            })

    # Apply filtering and transformation
    final_rows = filter_and_transform_rows(rows)

    # Write CSV
    fieldnames = [
        "page_number", "is_bold", "is_italic", "is_underline", "is_all_caps",
        "word_count", "norm_font_size", "font", "font_size", "x0", "y0", "x1", "y1",
        "space_above", "space_below", "space_left", "space_right", "text"
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

def process_folder(input_folder, output_folder):
    """Processes all PDF files in a given folder."""
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
