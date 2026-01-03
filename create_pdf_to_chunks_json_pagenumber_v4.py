
import fitz  # PyMuPDF for PDF parsing
import re
import os
import sys
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fuzzywuzzy import fuzz  # Fuzzy string matching

# Load PDF

# Define known headers and footers for fuzzy matching
HEADER_LINES = [
    "IEEE Std 802.11-2020",
    "IEEE ",
    "Standard for Information Technology—Local and Metropolitan Area Networks—Specific Requirements",
    "Part 11: Wireless LAN MAC and PHY Specifications",
    "Copyright © 2007 IEEE. All rights reserved.",
    "Restrictions apply",
    "Authorized licensed use limited to",
    "Copyright © 20211",
    "LOCAL AND METROPOLITAN AREA NETWORKS—SPECIFIC REQUIREMENTS",
    "Authorized licensed use limited to: rajan batra"
    "Authorized licensed use",
    "rajan batra",
    "Downloaded on December"
]

FOOTER_LINES = [
    "Copyright © 2007 IEEE. All rights reserved.",
    "Restrictions apply",
    "Authorized licensed use limited to",
    "rajan batra",
    "Downloaded on December",
    "Copyright © 20211",
    "IEEE Std 802.11-2020",
    "IEEE ",
    "Standard for Information Technology—Local and Metropolitan Area Networks—Specific Requirements",
    "Part 11: Wireless LAN MAC and PHY Specifications",
    "Copyright © 2007 IEEE. All rights reserved.",
    "Restrictions apply",
    "Authorized licensed use limited to",
    "Copyright © 20211",
    "LOCAL AND METROPOLITAN AREA NETWORKS—SPECIFIC REQUIREMENTS",
    "Authorized licensed use limited to: rajan batra"
    "Authorized licensed use",
    "rajan batra",
    "Downloaded on December"
]

SIMILARITY_THRESHOLD = int(os.environ.get("SIMILARITY_THRESHOLD"))  # Fuzzy match threshold (0-100)

# Regex to detect sections & sub-sections inline (Handles major & minor sections)

#clause_pattern = re.compile(
#    r"^\s*(?:\d+\.\s+[A-Z].*|\d+(?:\.\d+)+\s+[A-Z].*)$",
#    re.MULTILINE
#)

annex_pattern = re.compile(r"^\s*Annex\s+([A-Z])\b")

clause_pattern = re.compile(
    r"^\s*(\d+\.\s+([A-Z].*|\d.*)|\d+(?:\.\d+)+\s+([A-Z].*|\d.*))$",
    re.MULTILINE
)


def is_valid_next_section(current_section_number, curr_key, next_key):
    """
    Checks if next_key is a valid subsequent section to curr_key.
    This improved version is more robust.
    """

    if not curr_key:
      if (next_key[0] == 1): 
          new_heading = 1
          return True 
     # else:
     #   print("next key none:", next_key)

    if not curr_key or not next_key:
        return False

    # A section cannot precede itself
    if next_key == curr_key:
        return False

    # Check for new child section (e.g., 5.1 -> 5.1.1)
    if next_key[:len(curr_key)] == curr_key and len(next_key) == len(curr_key) + 1:
        return next_key[-1] == 1

    # Find the point where the keys diverge
    mismatch_index = -1
    for i in range(min(len(curr_key), len(next_key))):
        if curr_key[i] != next_key[i]:
            mismatch_index = i
            break

    # If no mismatch is found, it's an invalid case (e.g., (1,2) to (1,2,3))
    # This is already handled by the child section check.
    if mismatch_index == -1:
        return False

    # The prefix must match up to the mismatch point
    if curr_key[:mismatch_index] != next_key[:mismatch_index]:
        return False

    # The next_key's digit at the mismatch point must be exactly one greater
    # than the curr_key's digit at the same point.
    if next_key[mismatch_index] != curr_key[mismatch_index] + 1:
        return False

    # All subsequent digits in next_key must be 1 (e.g., for a jump from 5.1.2 to 5.2.1)
    for i in range(mismatch_index + 1, len(next_key)):
        if next_key[i] != 1:
            return False

    # All checks passed, it's a valid transition.
    return True

# Storage for extracted sections
sections = []
next_key = None
curr_key = None
current_section = None  # Tracks ongoing section
current_title = None
new_heading = 0
current_section_number = None
current_page_start = None

annex_page_number = int(os.environ.get("SPEC_ANNEX_START_PAGE_NUMBER", "0"))
removed_lines = set()  # Track removed headers & footers

print("\n🔹 **read the starting chunk number**")


# Regex: line starting with optional spaces, then "Annex ", then one capital A-Z

#  pdf_path = sys.argv[1]
pdf_path = os.environ.get("TRIM_SPEC_WIFI")
doc = fitz.open(pdf_path)

def read_chunk_counter(filename):
    """Reads the starting chunk number from a file."""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                print("Invalid chunk counter file. Resetting to 1.")
                return 1
    return 1

def section_key(s: str | None):
    if not s:  # catches None, '' etc.
        return ()
    try:
        # Filter out empty strings before mapping to int
        parts = [part for part in s.split('.') if part]
        return tuple(map(int, parts))
    except ValueError:
        return ()

def save_chunk_counter(chunk_id, filename="chunk_counter.txt"):
    """Saves the last allocated chunk ID to a file."""
    with open(filename, "w") as f:
        f.write(str(chunk_id))

#print("\n🔹 **Processing PDF: Removing Headers/Footers & Handling Inline Sections**")

    
page_text = doc[0].get_text("text")
lines = page_text.split("\n")
annex_match = 0
annex_mode = 0 
top_line = lines[3]  # Get the `n`-th line from the top
# Match a standalone number (1 to 4 digits)
#match = re.search(r"\b\d{1,4}\b", top_line)
match = 0
#if match:
  #extracted_page_number = int(match.group())
  #current_page_start = extracted_page_number 
  #current_section_page_start = current_page_start
  #print("extracted page number", extracted_page_number, current_page_start)

current_page_start = 1 
current_section_page_start = current_page_start

for page_num in range(len(doc)):
    page_text = doc[page_num].get_text("text")
    lines = page_text.split("\n")

#    top_line = lines[3]  # Get the `n`-th line from the top

    # Match a standalone number (1 to 4 digits)
#    match = re.search(r"\b\d{1,4}\b", top_line)
#    if match:
#      extracted_page_number = int(match.group())

    # Detect and remove headers/footers
    cleaned_lines = []
    for i, line in enumerate(lines):
        #if(i<5):
        #  print (line)
        is_header = i < 5 and any(fuzz.ratio(line.strip(), header) >= SIMILARITY_THRESHOLD for header in HEADER_LINES)
        is_footer = i >= len(lines) - 5 and any(fuzz.ratio(line.strip(), footer) >= SIMILARITY_THRESHOLD for footer in FOOTER_LINES)

        if is_header or is_footer:
            removed_lines.add(line.strip())
            #print(f"🛑 Removed Header/Footer on Page {page_num + 1}: {line.strip()}")
        else:
            cleaned_lines.append(line)

    page_text = "\n".join(cleaned_lines).strip()

    # Process each line and detect inline section headers
    for line in cleaned_lines:
        line = line.strip()

        new_heading = 0
        #print("line", line)
        if clause_pattern.match(line) and not annex_mode:  # Found a new section or sub-section
          parts = line.split()
          next_section_number = parts[0] # Extract "11.2.3.5.1" from "11.2.3.5.1 Sub-section Title"
          next_key = section_key(next_section_number)
          curr_key = section_key(current_section_number)

          #print("\n next key", next_key);
          #print("\n curr key", curr_key);
          #print("\n curr section number", current_section_number);

          if is_valid_next_section (current_section_number, curr_key, next_key):
            new_heading = 1
            #print("xx new heading")

        if current_page_start >= annex_page_number:
          annex_mode = 1
          annex_match = annex_pattern.match(line)  # Found a new section or sub-section
          if annex_match:  # Found a new section or sub-section
            annex_letter = annex_match.group(1)       # Extract the letter, e.g., "A"
            next_section_number = annex_letter
            annex_title = f"Annex {annex_letter}"      # Section title will be "Annex X"
            new_heading = 1
        # Save current section if it exists
            #print("found annex", annex_title)
                    
        if new_heading == 1 and (annex_match or clause_pattern.match(line)):  # Found a new section or sub-section
          if current_section is not None:
              sections.append({
                   "section_number": current_section_number,
                   "title": current_title,
                   "content": current_section.strip(),
                   "page_start": current_section_page_start,
                   "page_end": current_page_start
                   })
              len_sections = len(sections)
#              print(f"✅ Finalized Section: {len_sections}- {current_section_number} ({current_title}) - Pages {current_section_page_start} to {current_page_start}")
     
                 # Start a new section or sub-section
          if (annex_match):
            current_section_number = next_section_number # Extract Annex letter
            current_title = annex_title 
          else:
            current_section_number = line.split()[0]  # Extract "11.2.3.5.1" from "11.2.3.5.1 Sub-section Title"

#          current_title = line  # Full section title
            parts = line.split(' ', 1)

          # The second element (index 1) will be the rest of the string

            if len(parts) > 1:
              current_title = parts[1]
            else:
              current_title = ""

          current_section = ""  # Reset content
          current_section_page_start = current_page_start  # Reset content
 
          len_sections = len(sections)
          #print(f"📌 New Section Detected: len_sections - {current_section_number} ({current_title}) - Starting at Page {current_page_start}")

        # Append content to the current section
        if current_section is None:
            current_section = line  # If it's the first detected section
        else:
            current_section += "\n" + line  # Append content inline, even if it's an immediate sub-section

    current_page_start += 1  # Store starting page

# Save the last section after the loop ends
if current_section:
    sections.append({
        "section_number": current_section_number,
        "title": current_title,
        "content": current_section.strip(),
        "page_start": current_section_page_start,
        #"page_end": len(doc)  # Ends at the last page
        "page_end": current_page_start  # Ends at the last page
    })

    len_sections = len(sections)
    #print(f"✅ Finalized Last Section: {len_sections} - {current_section_number} ({current_title}) - Pages {current_section_page_start} to {current_page_start}")

print("no of sections: \n", len(sections))

# Print Removed Headers & Footers at the End
#print("\n🔹 **Summary of Removed Headers & Footers:**")
#for line in sorted(removed_lines):
#    print(f"- {line}")

# Step 3: Chunking with RecursiveCharacterTextSplitter
# chunk_size = 900000 
# chunk_overlap = 500
chunk_size = int(os.environ.get("CHUNK_SIZE"))
chunk_overlap = int(os.environ.get("CHUNK_OVERLAP"))

initial_chunks = []
chunk_id_counter = os.environ.get("CHUNK_ID_COUNTER")
chunk_counter = read_chunk_counter(chunk_id_counter)  # Unique chunk ID counter

print("\n🔹 **Chunking Process Started**", chunk_counter)
for section in sections:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(section["content"])
    
    for i, chunk in enumerate(chunks):
        page_start = section.get("page_start", 1)  # Default to 1 if missing
        section_number = section.get("section_number", "UNKNOWN")
        section_title = section["title"]

        # Debugging: Check if values are None
        if page_start is None:
            print(f"⚠️ Warning: page_start is None for section: {section_title}")
        if section_number is None:
            print(f"⚠️ Warning: section_number is None for section: {section_title}")

        #chunkid = f"{chunk_counter:03d}-{page_start:02X}-{section_number}"
        #chunkid = f"{chunk_counter:03d}-{(page_start if page_start is not None else 1):02X}-{section_number or 'UNKNOWN'}"
        chunkid = f"{chunk_counter:03d}"


        #print(f"✅ Created Chunk: {chunkid} (Pages {section['page_start']} to {section['page_end']})")
        #print(f"   🔹 Content Preview: {chunk[:150]}...")  # Print first 150 characters

        initial_chunks.append({
            "chunkid": chunkid,
            "section_number": section_number,
            "section_title": section_title,
            "page_start": page_start,
            "page_end": section["page_end"],
            "size": len(chunk),
            "content": chunk
        })

        chunk_counter += 1  # Increment unique chunk ID

# Step 4: Save Data to JSON
#chunk_file = f"{os.path.splitext(pdf_path)[0]}-chunks.json"
chunk_file = os.environ.get("SPEC_CHUNKS")


with open(chunk_file, "w", encoding="utf-8") as f:
    json.dump(initial_chunks, f, indent=4)

save_chunk_counter(chunk_counter)

# Print Summary
print("\n🔹 **Processing Complete**")
print(f"🔹 JSON Output File: {chunk_file}")
print(f"🔹 Total sections processed: {len(sections)}")
print(f"🔹 Total chunk count: {len(initial_chunks)}")





