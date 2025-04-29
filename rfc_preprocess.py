'''
This file is to preprocess the document of RFC files
'''

import re
import json
import os

import tiktoken

# A small helper to count tokens with the same encoding your LLM uses.
_encoding = tiktoken.get_encoding("cl100k_base")  # or your model’s encoding

def count_tokens(text: str) -> int:
    """Return exact token length of text under cl100k_base."""
    return len(_encoding.encode(text))


def clean_rfc_headers(text: str) -> str:
    """
    Cleans an RFC text file by removing header lines.
    This version replaces form feed characters (\f) with newline characters (\n)
    and then processes each line to remove headers if at least two header patterns match.
    """
    # Replace form feed characters with newline to ensure consistent line separation.
    text = text.replace('\f', '\n')
    
    # Define multiple regex patterns for common header artifacts.
    header_patterns = [
        r'\[Page\s*\d+\]',                       # Matches: [Page 13]
        r'\bRFC\s*\d+\b',                         # Matches: RFC 7826
        r'\bStandards\s+Track\b',                 # Matches: Standards Track
        r'\bRTSP\s+\d+\.\d+\b',                    # Matches: RTSP 2.0
        # Matches dates in the format "December 2016" etc.
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
    ]
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in header_patterns]

    # Split text into lines.
    lines = text.splitlines()
    cleaned_lines = []
    
    # Process each line individually.
    for line in lines:
        stripped_line = line.strip()
        # Count the number of header patterns that match the line.
        match_count = sum(1 for cp in compiled_patterns if cp.search(stripped_line))
        # Remove the line if at least two header patterns match.
        if match_count >= 2:
            continue
        cleaned_lines.append(line)
    
    # Reassemble the cleaned lines into a single string.
    cleaned_text = "\n".join(cleaned_lines)
    
    # Collapse multiple consecutive newlines into a single newline.
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    
    return cleaned_text


def extract_section_from_toc_line(line: str):
    """
    Extracts the section number and section name from a TOC line.
    
    This function supports several formats:
      - Numeric entries: "22. IANA Considerations ..........................................217"
      - Appendix entries with prefix: "Appendix A. Examples .............................................248"
      - Appendix level-two: "A.1. Media on Demand (Unicast) ................................248"
      
    The regex works as follows:
      - Optionally match "Appendix" followed by whitespace (captured in group 1).
      - Capture a section number that starts with either a letter or digits, 
        optionally followed by a dot and digits, and ends with a dot (group 2).
      - After some whitespace, capture the section title (group 3) up to the leader dots.
      - Leader dots (at least two) and a trailing page number are then matched.
      
    Returns a tuple (section_number, section_name) where:
      - If the "Appendix" prefix was present, the section number is returned as "Appendix <number>".
      - Otherwise, just the captured section number.
      
    If the line does not match the pattern, returns None.
    """
    pattern = re.compile(r'''
        ^\s*
        (?:(Appendix)\s+)?               # Optional "Appendix" prefix in group 1
        ((?:[A-Z]|\d+)(?:\.[0-9]+)?\.)    # Section number in group 2, e.g. "22." or "A.1."
        \s+
        (.+?)                           # Section title in group 3 (non-greedy)
        \s+\.{2,}\s*\d+\s*$             # Leader dots and trailing page number
    ''', re.IGNORECASE | re.VERBOSE)
    
    match = pattern.search(line)
    if match:
        prefix = match.group(1)
        section_number = match.group(2)
        section_title = match.group(3).strip()
        # If there's an Appendix prefix, prepend it to the section number.
        if prefix:
            section_number = f"Appendix {section_number}"
        return section_number, section_title
    else:
        return None
    

def process_toc_file(input_file: str, output_file: str):
    """
    Processes the file line by line to extract TOC entries (level-two only) and
    removes those lines from the file content.
    
    Writes the updated content to 'output_file' and returns a list of tuples:
        [(section_number, section_name), ...]
    
    For example, if a line is:
       "17.1. Informational 1xx .......................................113"
    then the tuple ("17.1.", "Informational 1xx") is returned and that line is removed.
    """
    extracted_sections = []
    updated_lines = []
    
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    
    for line in lines:
        result = extract_section_from_toc_line(line)
        if result is not None:
            # We found a TOC line; record its section number and name.
            extracted_sections.append(result)
        else:
            # Not a TOC line; keep it in the updated content.
            updated_lines.append(line)
    
    # Write the updated content (without TOC lines) to the output file.
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(updated_lines)
    
    return extracted_sections



# needed for both level-1 and level-2 splits.
def split_document_by_sections(sections, document_text):
    """
    Original function you provided: splits on *exact* normalized headings 
    in `sections = [(sec_num, sec_name), …]`, returning
      [{"section": (sec_num, sec_name), "tag": "...", "content": "..."}…].
    """
    def normalize_heading_line(line: str) -> str:
        return " ".join(line.strip().split())

    # Prepare normalized candidates
    normalized = []
    for num, name in sections:
        cand = normalize_heading_line(f"{num} {name}")
        normalized.append((num, name, cand))

    lines = document_text.splitlines()
    segments = []
    current = {"section": None, "tag": None, "content": []}
    stack = []

    for line in lines:
        norm = normalize_heading_line(line)
        match = next(
            ((n, nm) for (n, nm, cand) in normalized if norm == cand),
            None
        )
        if match:
            # finish previous
            if current["section"]:
                current["content"] = "\n".join(current["content"]).strip()
                segments.append(current)
            sec_num, sec_name = match
            level = len(sec_num.rstrip(".").split("."))

            # maintain hierarchy stack
            while stack and len(stack) >= level:
                stack.pop()
            stack.append((sec_num, sec_name, level))

            hier = " ".join(item[1] for item in stack)
            tag = f"Section {sec_num} {hier}"
            current = {"section": (sec_num, sec_name), "tag": tag, "content": []}
        else:
            if current["section"]:
                current["content"].append(line)

    # last one
    if current["section"]:
        current["content"] = "\n".join(current["content"]).strip()
        segments.append(current)

    return segments



# The new top-level orchestrator
def split_within_token_limit(
    sections,
    document_text,
    max_tokens: int = 40_000
):
    """
    1) Split into level-1 sections.
    2) If a chunk ≤ max_tokens 👉 keep.
       Else 👉 split on level-2, then greedily re-combine until each
       combined piece ≤ max_tokens.
    Returns the final list of {"section", "tag", "content"} dicts.
    """
    # helper to get “level” from a sec_num string
    def level_of(num: str) -> int:
        return len(num.rstrip(".").split("."))

    # pick out level-1 headings
    lvl1 = [(n, nm) for n, nm in sections if level_of(n) == 1]
    lvl1_segments = split_document_by_sections(lvl1, document_text)

    final = []
    for seg in lvl1_segments:
        toklen = count_tokens(seg["content"])
        if toklen <= max_tokens:
            final.append(seg)
            continue

        # over-limit: try to split on level-2 under this parent
        parent_num, parent_name = seg["section"]
        lvl2 = [
            (n, nm) for n, nm in sections
            if level_of(n) == 2 and n.startswith(parent_num)
        ]

        # if no lvl-2, give up and keep as is
        if not lvl2:
            final.append(seg)
            continue

        children = split_document_by_sections(lvl2, seg["content"])
        # greedy combine
        i = 0
        while i < len(children):
            curr = children[i]
            combo_content = curr["content"]
            combo_tokens  = count_tokens(combo_content)
            nums = [curr["section"][0]]
            names= [curr["section"][1]]
            j = i + 1

            while j < len(children):
                nxt = children[j]
                ntok = count_tokens(nxt["content"])
                if combo_tokens + ntok <= max_tokens:
                    combo_content += "\n" + nxt["content"]
                    combo_tokens  += ntok
                    nums.append(nxt["section"][0])
                    names.append(nxt["section"][1])
                    j += 1
                else:
                    break

            # build combined tag:
            subparts = " ".join(f"{num} {nm}" for num, nm in zip(nums, names))
            tag = f"Section {parent_num} {parent_name} {subparts}"
            final.append({
                "section": (parent_num, parent_name),
                "tag": tag,
                "content": combo_content
            })
            i = j

    return final


if __name__ == "__main__":
    # the protocol list
    # dirs = ["DCCP", "DHCP", "FTP", "IMAP", 
    #         "NNTP", "POP3", "RTSP", "SIP", "SMTP", "TCP"]
    
    
    '''---------------clean RFC file------------------------------'''
    # for prot in dirs:
    #     # check directory exists
    #     if not os.path.isdir(prot):
    #         print(f"Skipping '{prot}': not a directory.")
    #         continue

    #     # find all *_raw.txt files in this directory
    #     raw_files = [f for f in os.listdir(prot) if f.endswith("_raw.txt")]
    #     if not raw_files:
    #         print(f"No '_raw.txt' files found in '{prot}/'.")
    #         continue

    #     # process each raw file
    #     for raw_fn in raw_files:
    #         raw_path = os.path.join(prot, raw_fn)
    #         with open(raw_path, "r", encoding="utf-8") as f_in:
    #             raw_text = f_in.read()

    #         clean_text = clean_rfc_headers(raw_text)

    #         # build cleaned filename by swapping suffix
    #         cleaned_fn  = raw_fn.replace("_raw.txt", "_cleaned.txt")
    #         cleaned_path = os.path.join(prot, cleaned_fn)

    #         with open(cleaned_path, "w", encoding="utf-8") as f_out:
    #             f_out.write(clean_text)

    #         print(f"{prot}/{raw_fn} → {prot}/{cleaned_fn}")
    
    '''---------------extract the sections------------------------------'''
    dirs = ["DCCP"]
    for prot in dirs:
        # check directory exists
        if not os.path.isdir(prot):
            print(f"Skipping '{prot}': not a directory.")
            continue

        # find all *_cleaned.txt files in this directory
        cleaned_files = [f for f in os.listdir(prot) if f.endswith("_cleaned.txt")]
        if not cleaned_files:
            print(f"No '_cleaned.txt' files found in '{prot}/'.")
            continue
        
        #no_toc_files = [f for f in os.listdir(prot) if f.endswith("_no_toc.txt")]
    
        # process each cleaned file
        for cleaned_fn in cleaned_files:
            # Load the cleaned RFC file.
            input_file = os.path.join(prot, cleaned_fn)      # The original file with TOC lines.
            output_file  = input_file.replace("_cleaned.txt", "_no_toc.txt")
            
            # Process the file, extract TOC sections, and write the cleaned file.
            sections = process_toc_file(input_file, output_file)
            
            # Print the number of extracted sections and their details.
            print(f"Extracted {len(sections)} TOC entries (level-two sections):")
            for section_number, section_name in sections:
                print(f"  Section Number: {section_number}, Section Name: {section_name}")
            
            print(f"\nThe updated file without TOC lines has been saved as '{output_file}'.")
            
            with open(output_file) as f:
                text = f.read()
            
            segments = split_within_token_limit(sections, text, max_tokens=40000)
            
            for seg in segments:
                print(seg["tag"], "→", count_tokens(seg["content"]), "tokens\n","the first 20 chars content:", seg["content"][:20])
            
            
            
    