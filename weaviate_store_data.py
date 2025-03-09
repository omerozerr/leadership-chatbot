import os
import json
import glob
import weaviate
import weaviate.classes.config as wc
from weaviate.classes.init import Auth

from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env in the project root

# --- Helper function to split text into eight parts ---
def split_text_into_eight(text):
    n = 8
    chunk_size = len(text) // n
    parts = []
    for i in range(n):
        if i == n - 1:
            parts.append(text[i * chunk_size:])
        else:
            parts.append(text[i * chunk_size:(i + 1) * chunk_size])
    return parts

headers = {
    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
}

# Instantiate the client using the Weaviate Cloud URL and API key for admin
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WCD_DEMO_URL"),
    auth_credentials=Auth.api_key(os.getenv("WCD_DEMO_ADMIN_KEY")),
    headers=headers,
)
client.collections.delete_all()

# Create a collection for Transcript objects using the new collection API.
client.collections.create(
    name="Transcript",
    properties=[
        wc.Property(
            name="video_url",
            data_type=wc.DataType.TEXT,
            description="The YouTube video URL."
        ),
        wc.Property(
            name="text",
            data_type=wc.DataType.TEXT,
            description="The transcript text."
        )
    ],
    vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
    generative_config=wc.Configure.Generative.openai()
)
print("Collection 'Transcript' created.")

# Directory containing the JSON files (each with video_url and text fields)
json_folder = "transc_single_jsons"

Transcript = client.collections.get("Transcript")
# Loop through each JSON file and add it as an object to the Transcript collection
for filepath in glob.glob(os.path.join(json_folder, "*.json")):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Each JSON is expected to have "video_url" and "text" fields.
    video_url = data.get("video_url", "Unknown")
    text = data.get("text", "No transcript text available.")
    # Split the transcript text into eight parts
    parts = split_text_into_eight(text)
    for part in parts:
        Transcript.data.insert(
            properties={
                "video_url": video_url,
                "text": part
            }
        )
    print(f"Added object from {os.path.basename(filepath)}")

# Print meta information to verify the data in Weaviate
metainfo = client.get_meta()
print(json.dumps(metainfo, indent=2))

client.close()
