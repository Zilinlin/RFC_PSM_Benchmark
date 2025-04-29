'''
This file is to preprocess the document of RFC files
'''

import re
import json
import os


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


if __name__ == "__main__":
    # the protocol list
    dirs = ["DCCP", "DHCP", "FTP", "IMAP", 
            "NNTP", "POP3", "RTSP", "SIP", "SMTP", "TCP"]
    
    
    for prot in dirs:
        # check directory exists
        if not os.path.isdir(prot):
            print(f"Skipping '{prot}': not a directory.")
            continue

        # find all *_raw.txt files in this directory
        raw_files = [f for f in os.listdir(prot) if f.endswith("_raw.txt")]
        if not raw_files:
            print(f"No '_raw.txt' files found in '{prot}/'.")
            continue

        # process each raw file
        for raw_fn in raw_files:
            raw_path = os.path.join(prot, raw_fn)
            with open(raw_path, "r", encoding="utf-8") as f_in:
                raw_text = f_in.read()

            clean_text = clean_rfc_headers(raw_text)

            # build cleaned filename by swapping suffix
            cleaned_fn  = raw_fn.replace("_raw.txt", "_cleaned.txt")
            cleaned_path = os.path.join(prot, cleaned_fn)

            with open(cleaned_path, "w", encoding="utf-8") as f_out:
                f_out.write(clean_text)

            print(f"{prot}/{raw_fn} → {prot}/{cleaned_fn}")
    
    