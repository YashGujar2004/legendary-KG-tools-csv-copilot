import pdfplumber
import os
import json
import re

input_pdf_path = os.environ.get("FULL_SPEC_WIFI")
output_file_path = os.environ.get("VAR_TABLE_LIST")

def extract_tables_and_captions(pdf_path):
    """
    Extracts tables and their likely captions from a PDF file with specific rules.

    This function iterates through each page of a PDF, identifies tables, and
    searches defined regions both above and below each table's bounding box.
    It returns a list of dictionaries for tables where a caption is found.
    The final caption is set based on a set of rules:
    1. Select the caption if "Figure" or "Table" keyword is found.
    2. If a caption is found in the below bounding box and starts with "Figure",
       it is trimmed by starting with the "Figure" keyword.
    3. If no caption is found for a table, the first caption found on the rest of the page
       is assigned to it. If no other captions are found, "NO CAPTION FOUND" is assigned.
    4. The script will only check the 'below' bounding box for captions if the
       table's data consists of only a single row (one list of data).
    5. It will only check 'above' for captions if the table has multiple rows.
    6. NEW: If a multi-row table has no caption in the 'above' bounding box,
       it will then search a smaller 'below' bounding box (70 pixels in height).
    7. NEW: If a caption is still not found for a single-row table after the initial search,
       the script will check a larger 'below' bounding box (70 pixels in height).

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        list: A list of dictionaries, where each dictionary contains the extracted
              caption text and the table data for a single table found in the PDF.
    """
    extracted_data = []
    temp_storage = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Opened PDF: {pdf_path}")
            for i, page in enumerate(pdf.pages):
                #print(f"Processing Page {i + 1}...")

                # Reset temporary storage for each new page
                temp_storage = []

                # Step 1: Identify all tables on the current page
                page_tables = page.find_tables()
                if not page_tables:
                    #print(f"No tables found on Page {i + 1}.")
                    continue

                #print(f"{len(page_tables)} table(s) found on Page {i + 1}.")

                for j, table_obj in enumerate(page_tables):
                    table_bbox = table_obj.bbox
                    table_data = table_obj.extract()

                    # Step 2: Define regions of interest (ROIs) for the caption
                    above_caption_text = None
                    below_caption_text = None

                    # Check if the table data is a single row
                    if len(table_data) == 1:
                        # Only search below for single-row tables
                        #print(f"  -- Table {j + 1} is a single row, checking below only.")
                        below_roi = (
                            0,
                            table_bbox[3],
                            page.width,
                            table_bbox[3] + 40
                        )
                        below_caption_text = page.crop(below_roi).extract_text()
                    else:
                        # Only search above for multi-row tables
                        #print(f"  -- Table {j + 1} is multi-row, checking above only.")
                        above_roi = (
                            0,
                            table_bbox[1] - 50,
                            page.width,
                            table_bbox[1]
                        )
                        if above_roi[1] < 0:
                            above_roi = (above_roi[0], 0, above_roi[2], above_roi[3])
                        above_caption_text = page.crop(above_roi).extract_text()

                    # Step 4: Check captions based on your rules
                    final_caption = None

                    # Rule 1: Check above caption first (if available for multi-row tables)
                    if above_caption_text:
                        lower_text = above_caption_text.lower()
                        if "table" in lower_text:
                            start_index = lower_text.find("table")
                            final_caption = above_caption_text[start_index:].strip()

                    # Check below caption based on original rule for single-row tables
                    if not final_caption and len(table_data) == 1:
                        if below_caption_text:
                            lower_text = below_caption_text.lower()
                            if "figure" in lower_text:
                                start_index = lower_text.find("figure")
                                final_caption = below_caption_text[start_index:].strip()

                    # NEW RULE: Expanded search below for single-row tables if no caption found
                    if not final_caption and len(table_data) == 1:
                        #print(f"  -- No caption found for single-row table. Expanding search to 70px region below table {j + 1}.")
                        expanded_below_roi = (
                            0,
                            table_bbox[3],
                            page.width,
                            table_bbox[3] + 70
                        )
                        expanded_below_caption_text = page.crop(expanded_below_roi).extract_text()
                        if expanded_below_caption_text:
                            lower_text = expanded_below_caption_text.lower()
                            if "figure" in lower_text:
                                start_index = lower_text.find("figure")
                                final_caption = expanded_below_caption_text[start_index:].strip()
                            # elif "table" in lower_text:
                            #     start_index = lower_text.find("table")
                            #     final_caption = expanded_below_caption_text[start_index:].strip()

                    # Rule 6: Check a smaller below caption for multi-row tables if no above caption
                    if not final_caption and len(table_data) > 1:
                        #print(f"  -- No caption found above. Checking a 70px region below table {j + 1}.")
                        below_roi_multi = (
                            0,
                            table_bbox[3],
                            page.width,
                            table_bbox[3] + 70
                        )
                        below_caption_text = page.crop(below_roi_multi).extract_text()
                        if below_caption_text:
                            lower_text = below_caption_text.lower()
                            if "figure" in lower_text:
                                start_index = lower_text.find("figure")
                                final_caption = below_caption_text[start_index:].strip()


                    # Append to temporary storage, handling no-caption case
                    temp_storage.append({
                        "page_number": i + 1,
                        "caption": final_caption,
                        "table_data": table_data
                    })
                    #print(f"\n--- Table {j + 1} on Page {i + 1} ---")
                    if final_caption:
                        print(f"Caption: {final_caption}")
                    else:
                        print("No caption with 'Figure' or 'Table' keyword found.")

                # Rule 3: Post-processing to assign missing captions
                all_captions_on_page = [item["caption"] for item in temp_storage if item["caption"]]

                for item in temp_storage:
                    if not item["caption"]:
                        assigned = False
                        # Find the next available caption on the page
                        try:
                            # Get index of the current item in the temp_storage
                            current_index = temp_storage.index(item)

                            # Search for a caption in the rest of the list
                            for next_item in temp_storage[current_index + 1:]:
                                if next_item["caption"]:
                                    item["caption"] = next_item["caption"]
                                    assigned = True
                                    #print(f"  -- Assigned next available caption to a previously un-captioned table --")
                                    break
                        except ValueError:
                            pass

                        if not assigned:
                            item["caption"] = "NO CAPTION FOUND"
                            #print("  -- No other caption found on the page; assigned 'NO CAPTION FOUND' --")

                # Add the processed data from this page to the main list
                extracted_data.extend(temp_storage)

    except FileNotFoundError:
        print(f"Error: The file at path '{pdf_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return extracted_data

# /n REMOVAL NEWLINES FROM JSON

def replace_newlines(obj):
    if isinstance(obj, str):
        return obj.replace("\n", " ")
    elif isinstance(obj, list):
        return [replace_newlines(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: replace_newlines(value) for key, value in obj.items()}
    return obj

if __name__ == "__main__":

    # Make sure the PDF file is in the same directory as this script.
    tables_with_captions = extract_tables_and_captions(input_pdf_path)

    if tables_with_captions:

        try:
            # Write the list of dictionaries to a JSON file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(tables_with_captions, f, ensure_ascii=False, indent=4)
            print(f"\nSuccessfully saved the extracted data to {output_file_path}")
            
            # clean for /n
            
            with open(output_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cleaned_data = replace_newlines(data)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=4)        

        except Exception as e:
            print(f"\nError writing to JSON file: {e}")
    else:
        print("\nNo data was extracted or no captions matched the keywords. Please check the file path and PDF content.")
        
