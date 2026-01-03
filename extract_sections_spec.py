import fitz
import os

def extract_pages(input_pdf, output_pdf, start_page, end_page):
    """
    Extracts a range of pages from a PDF and saves them to a new PDF.

    Args:
        input_pdf: Path to the input PDF file.
        output_pdf: Path to save the extracted pages.
        start_page: The starting page number (1-based index).
        end_page: The ending page number (1-based index).
    
    Returns:
        True on success, False on failure. Prints error messages to console.
    """
    try:
        doc = fitz.open(input_pdf)
        new_doc = fitz.open()

        if not 1 <= start_page <= end_page <= doc.page_count:
            print("Invalid page range. Please provide valid page numbers.")
            return False

        for page_num in range(start_page - 1, end_page):  # Page numbers are 0-based in fitz
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        new_doc.save(output_pdf)
        return True

    except fitz.FileDataError:
        print("Invalid input PDF file.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Example Usage
input_pdf_path = os.environ["SPEC"]
output_pdf_path = os.environ["TRIM_SPEC_WIFI"] # Replace with desired output path
# start_page_num = 219 #Extract from page 2
# end_page_num = 3429 # Extract up to and including page 5
start_page_num = int(os.environ["START_PAGE_NUM"])
end_page_num = int(os.environ["END_PAGE_NUM"])


if extract_pages(input_pdf_path, output_pdf_path, start_page_num, end_page_num):
    print(f"Pages {start_page_num} to {end_page_num} extracted successfully to {output_pdf_path}")
else:
    print("Page extraction failed.")
