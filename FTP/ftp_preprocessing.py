'''
Because the FPT documentation is different from the other documentations,
we design a seperate documentation preprocessing script for it.

'''

import re
import json

def clean_ftp_page_titles(text: str) -> str:
    """
    1) Normalize Windows line endings to '\n'
    2) Split on form-feed (\f)
    3) For each page:
       - Drop any line matching our header/footer patterns
       - Also drop blank lines until the first real content line
    4) Re-join pages with a single '\f'
    """
    # 1) Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 2) Patterns to remove
    page_num_re   = re.compile(r'\[\s*Page\s*\d+\s*\]')
    rfc_header_re = re.compile(r'^RFC\s+959.*\d{4}$')
    title_re      = re.compile(r'^File\s+Transfer\s+Protocol$')
    author_re     = re.compile(r'Postel\s*&\s*Reynolds')

    pages = text.split('\f')
    cleaned_pages = []

    for page in pages:
        lines = page.split('\n')
        new_lines = []
        seen_content = False

        for line in lines:
            # 3a) Always drop header/footer lines
            if ( page_num_re.search(line)
              or rfc_header_re.match(line)
              or title_re.match(line)
              or author_re.search(line)
            ):
                continue

            # 3b) Before first content, also drop blank lines
            if not seen_content:
                if not line.strip():  # pure-blank
                    continue
                seen_content = True

            # 3c) Once content started, keep everything (including blank lines)
            new_lines.append(line)

        cleaned_pages.append("\n".join(new_lines))

    # 4) stitch back together
    return "\f".join(cleaned_pages)




def extract_sections(text: str):
    """
    Split `text` into level-1 sections (1.–8.) and appendices (I, II),
    in the order they appear, but only one occurrence each.

    A heading must have TWO or more spaces after the number:
       2.  Overview          ← matches
       2. - Something        ← won't match (only one space before dash)

    Returns a list of dicts with keys:
      - section_number
      - section_name
      - tag
      - content
    """
    # Only match EXACTLY top-level headings: e.g. "2.  Overview"
    sec_re = re.compile(r'^\s*(\d+)\.\s{2,}(.+?)\s*$')
    # Match APPENDIX lines like "APPENDIX I - PAGE STRUCTURE"
    app_re = re.compile(r'^\s*APPENDIX\s+([IVX]+)\s*-\s*(.+?)\s*$', re.IGNORECASE)

    desired_secs = [f"{i}." for i in range(1, 9)]  # "1.", "2.", ..., "8."
    seen_secs    = set()
    seen_apps    = set()

    segments = []
    current = None

    for line in text.splitlines():
        # Try numeric section
        m = sec_re.match(line)
        # Try appendix
        n = app_re.match(line)

        if m:
            num, title = m.group(1), m.group(2).strip()
            num_dot = f"{num}."
            # Only start a new segment if it's in 1–8 and not yet seen
            if num_dot in desired_secs and num_dot not in seen_secs:
                seen_secs.add(num_dot)
                # close previous
                if current:
                    current["content"] = current["content"].rstrip("\n")
                    segments.append(current)
                # start new
                tag = f"Section {num_dot} {title}"
                current = {
                    "section_number": num_dot,
                    "section_name":   title,
                    "tag":            tag,
                    "content":        ""
                }
                continue

        elif n:
            roman, title = n.group(1).upper(), n.group(2).strip()
            app_id = f"APPENDIX {roman}"
            if app_id not in seen_apps:
                seen_apps.add(app_id)
                if current:
                    current["content"] = current["content"].rstrip("\n")
                    segments.append(current)
                tag = f"Section {app_id} {title}"
                current = {
                    "section_number": app_id,
                    "section_name":   title,
                    "tag":            tag,
                    "content":        ""
                }
                continue

        # Otherwise, if we're inside a segment, accumulate
        if current:
            current["content"] += line + "\n"

    # finalize last
    if current:
        current["content"] = current["content"].rstrip("\n")
        segments.append(current)

    return segments



if __name__ == "__main__":
    raw_file = "rfc959_raw.txt"
    cleaned_file = "rfc959_cleaned.txt"
    
    '''clean the raw file'''
    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    cleaned_text = clean_ftp_page_titles(raw_text)
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    segments = extract_sections(cleaned_text)
    for segment in segments:
        print(f"Section: {segment['section_number']} - {segment['section_name']}")
        print(f"Tag: {segment['tag']}")
        print("Content:")
        print(segment['content'][:30]) # Print first 30 characters of content
        print("\n" + "="*40 + "\n")
    
    segment_json_file = "rfc959_segments.json"
    with open(segment_json_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

        
    
    
        
        
    
    