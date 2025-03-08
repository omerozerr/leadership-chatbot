
#############################################
# Step 2: Create embeddings and perform semantic search
#############################################
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import openai
load_dotenv()
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load the CSV we just created
#df = pd.read_csv("output/embedded_transcripts.csv")
# For compatibility, we assign the transcript text column to 'combined'
#df["combined"] = df["combined"]

# Compute embeddings for each segment and save them in a column 'ada_embedding'
#df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
#df.to_csv('output/embedded_transcripts_with_embeddings.csv', index=False)

# Reload the CSV and convert the embeddings from string to np.array
df = pd.read_csv('output/embedded_transcripts_with_embeddings.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)


def search_reviews(df, product_description, n=3, pprint=True):
    embedding = get_embedding(product_description, model='text-embedding-3-small')
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    if pprint:
        for r in res.itertuples():
            print(f"{r.filename} (Part {r.chunk_index}) - Similarity: {r.similarities:.4f}")
            snippet = r.combined.replace("\n", " ") + "..."
            print(snippet)
            print()
    return res

# Example usage:
res = search_reviews(df, '2014 yılında siber risklerin, saldırıların dünya ekonomisine ', n=3)
