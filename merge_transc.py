import os
import re
import json

# Define input and output folders
input_folder = "transc"
output_folder = "transc_single_jsons"
os.makedirs(output_folder, exist_ok=True)

# Regex to extract the video URL (expects a line like: "Video URL: https://www.youtube.com/watch?v=0_gUjjDZJvk")
video_url_pattern = re.compile(r"Video URL:\s*(https://www\.youtube\.com/watch\?v=[\w-]+)")

# Regex to detect timestamp header lines (e.g., "[00:00 - 00:30]:")
timestamp_pattern = re.compile(r"^\[\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\]:\s*$")

# Process each .txt file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        video_url = None
        text_lines = []
        
        # Process each line: extract the video URL and skip timestamp headers.
        for line in lines:
            line = line.strip()
            # Try to extract the video URL if not already done.
            if not video_url:
                match = video_url_pattern.match(line)
                if match:
                    video_url = match.group(1)
                    continue  # Skip this line once URL is extracted.
            # Skip lines that match the timestamp pattern.
            if timestamp_pattern.match(line):
                continue
            # Skip empty lines
            if not line:
                continue
            # Otherwise, add the line to our transcript text.
            text_lines.append(line)
        
        # Merge all remaining lines into a single string.
        combined_text = " ".join(text_lines)
        
        # Create a JSON object with the video URL and merged text.
        data = {
            "video_url": video_url if video_url else "Unknown",
            "text": combined_text
        }
        
        # Use the video ID (extracted from the URL) as the JSON filename, or fallback to the original filename.
        if video_url:
            video_id = video_url.split("v=")[-1]
        else:
            video_id = os.path.splitext(filename)[0]
        
        output_filepath = os.path.join(output_folder, f"{video_id}.json")
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {filename} -> {output_filepath}")
