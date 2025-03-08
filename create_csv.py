#############################################
# Step 1: Process raw transcripts and create CSV of segments
#############################################

import pandas as pd
import numpy as np
from openai import OpenAI
import os
import re
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
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

#############################################
# Create embeddings to perform semantic search


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Load the CSV we just created
df = pd.read_csv("output/embedded_transcripts.csv")

# Compute embeddings for each segment and save them in a column 'ada_embedding'
df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv('output/embedded_transcripts_with_embeddings.csv', index=False)