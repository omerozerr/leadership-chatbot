#############################################
# Step 1: Process raw transcripts and create CSV of segments
#############################################

import os
import re
import pandas as pd

transcripts_folder = "transc"
segments_list = []

# Regex to match a time marker line like "[22:00 - 22:30]:"
pattern = re.compile(r"^\[\d{2}:\d{2} - \d{2}:\d{2}\]:\s*$")

for filename in os.listdir(transcripts_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(transcripts_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        current_header = None
        current_text_lines = []
        chunk_index = 0
        for line in lines:
            if pattern.match(line.strip()):
                # When a new header is encountered, save previous segment if exists
                if current_header is not None and current_text_lines:
                    chunk_index += 1
                    segment_text = current_header + "\n" + "".join(current_text_lines).strip()
                    segments_list.append({
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "combined": segment_text
                    })
                    current_text_lines = []
                current_header = line.strip()
            else:
                current_text_lines.append(line)
        # Save any remaining segment after file end
        if current_header is not None and current_text_lines:
            chunk_index += 1
            segment_text = current_header + "\n" + "".join(current_text_lines).strip()
            segments_list.append({
                "filename": filename,
                "chunk_index": chunk_index,
                "combined": segment_text
            })

# Create a DataFrame and save to CSV
df_segments = pd.DataFrame(segments_list)
os.makedirs("output", exist_ok=True)
df_segments.to_csv("output/embedded_transcripts.csv", index=False)
print(f"CSV created with {len(df_segments)} segments.")