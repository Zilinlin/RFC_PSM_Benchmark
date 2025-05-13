# Revised script for hardcoded directory list
import os
import glob
import json

def analyze_directory(dir_path):
    # Find segments JSON
    segments_files = glob.glob(os.path.join(dir_path, '*_segments.json'))
    if segments_files:
        with open(segments_files[0], 'r', encoding='utf-8') as f:
            segments = json.load(f)
        seg_count = len(segments)
    else:
        seg_count = 0

    # Find state machine JSON
    sm_files = glob.glob(os.path.join(dir_path, '*_state_machine.json'))
    if sm_files:
        with open(sm_files[0], 'r', encoding='utf-8') as f:
            sm = json.load(f)
        state_count = len(sm.get('states', []))
        trans_count = len(sm.get('transitions', []))
    else:
        state_count = 0
        trans_count = 0

    return seg_count, state_count, trans_count

def main():
    # Hardcode your directory list here:
    directories = ["DCCP","DHCP", "FTP", "IMAP", 
           "NNTP", "POP3", "RTSP", "SIP", "SMTP", "TCP",
           "MQTT", "PPP", "PPTP", "BGP"
    ]

    print("Directory,Segments,States,Transitions")
    for d in directories:
        seg_count, state_count, trans_count = analyze_directory(d)
        print(f"{d},{seg_count},{state_count},{trans_count}")

if __name__ == "__main__":
    main()

# Save this script as `analyze_fsm_hardcoded.py`
