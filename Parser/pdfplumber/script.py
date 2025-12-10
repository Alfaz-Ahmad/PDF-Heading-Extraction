import os
import sys
import csv
import pdfplumber
import re


def is_number(text):
    return bool(re.match(r"^\d+(\.\d+)?$", text.strip()))


def is_all_caps(text):
    return text.isupper() and len(text) > 1


def get_font_ranking(elements):
    """Ranks font sizes from largest to smallest."""
    sizes = sorted(set(e["font_size"] for e in elements), reverse=True)
    return {size: rank + 1 for rank, size in enumerate(sizes)}


def merge_text_elements(page):
    """Extracts text elements from pdfplumber page keeping text as-is."""
    elements = []
    for char in page.chars:
        elements.append({
            "text": char.get("text", ""),
            "font_name": char.get("fontname", ""),
            "font_size": char.get("size", 0),
            "x0": char.get("x0", 0),
            "y0": char.get("top", 0),
            "x1": char.get("x1", 0),
            "y1": char.get("bottom", 0),
            "is_bold": 1 if "Bold" in char.get("fontname", "") else 0,
            "is_italic": 1 if "Italic" in char.get("fontname", "") or "Oblique" in char.get("fontname", "") else 0,
            "is_underline": 0,  # pdfplumber doesn't provide underline info
            "is_all_caps": 1 if is_all_caps(char.get("text", "")) else 0
        })
    return elements


def aggregate_elements(elements, page_height, page_width, font_rank):
    """Aggregates characters into blocks (instead of per line) and calculates spacing."""
    if not elements:
        return []

    # Sort elements by y0 (top) then x0 (left) for reading order
    sorted_elements = sorted(elements, key=lambda e: (round(e["y0"], 1), e["x0"]))
    
    rows = []
    current_block = None

    for elem in sorted_elements:
        # Check if the current character is part of the same text block
        # Criteria: similar y position (within 5 units) and same font size
        if current_block and \
           abs(elem["y0"] - current_block['y0_avg']) < 5 and \
           elem['font_size'] == current_block['font_size']:
            
            # Check for a space between words
            if elem['x0'] > current_block['x1_max'] + 2:
                current_block['text'] += " "
            
            current_block['text'] += elem['text']
            current_block['chars'].append(elem)
            current_block['x1_max'] = max(current_block['x1_max'], elem['x1'])
            current_block['y1_max'] = max(current_block['y1_max'], elem['y1'])
            current_block['y0_avg'] = (current_block['y0_avg'] * (len(current_block['chars']) - 1) + elem['y0']) / len(current_block['chars'])

        else:
            # Start a new block
            if current_block:
                rows.append(build_row(current_block['text'], current_block['chars'], page_height, page_width, font_rank))
            
            current_block = {
                'text': elem['text'],
                'chars': [elem],
                'y0_avg': elem['y0'],
                'x0_min': elem['x0'],
                'x1_max': elem['x1'],
                'y1_max': elem['y1'],
                'font_size': elem['font_size']
            }

    if current_block:
        rows.append(build_row(current_block['text'], current_block['chars'], page_height, page_width, font_rank))

    return rows


def build_row(text, chars, page_height, page_width, font_rank):
    x0 = min(c["x0"] for c in chars)
    y0 = min(c["y0"] for c in chars)
    x1 = max(c["x1"] for c in chars)
    y1 = max(c["y1"] for c in chars)
    font_size = chars[0]["font_size"]

    return {
        "text": text.strip(),
        "font": chars[0]["font_name"],
        "font_size": font_size,
        "norm_font_size": font_rank.get(font_size, len(font_rank)),
        "is_bold": max(c["is_bold"] for c in chars),
        "is_italic": max(c["is_italic"] for c in chars),
        "is_underline": max(c["is_underline"] for c in chars),
        "is_all_caps": 1 if is_all_caps(text.strip()) else 0,
        "word_count": len(text.strip().split()),
        "x0": round(x0, 2), "y0": round(y0, 2),
        "x1": round(x1, 2), "y1": round(y1, 2),
        "space_above": round(y0, 2),
        "space_below": round(page_height - y1, 2),
        "space_left": round(x0, 2),
        "space_right": round(page_width - x1, 2)
    }


def filter_and_transform_rows(rows):
    filtered_rows = []
    for row in rows:
        text = row['text']
        word_count = row['word_count']
        is_heading = row['is_all_caps']

        if is_number(text):
            continue
        if word_count <= 2 and not is_heading:
            continue

        filtered_rows.append(row)
    return filtered_rows


def process_pdf(pdf_path, output_csv):
    rows = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_height = page.height
                page_width = page.width

                elements = merge_text_elements(page)
                font_rank = get_font_ranking(elements)
                page_rows = aggregate_elements(elements, page_height, page_width, font_rank)

                for r in page_rows:
                    r['page_number'] = page_num

                rows.extend(page_rows)

        final_rows = filter_and_transform_rows(rows)

        fieldnames = [
            "page_number", "is_bold", "is_italic", "is_underline", "is_all_caps",
            "word_count", "norm_font_size", "font", "font_size", "x0", "y0", "x1", "y1",
            "space_above", "space_below", "space_left", "space_right", "text"
        ]

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)
    except Exception as e:
        print(f"An error occurred while processing {pdf_path}: {e}")
        
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