import os
import json
import numpy as np
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from elevenlabs.client import ElevenLabs  # Official ElevenLabs SDK
from elevenlabs import play
import openai  # For embedding calls
import pandas as pd
import time
import re
import weaviate
import weaviate.classes.query as wq
from weaviate.classes.init import Auth

load_dotenv()  # Loads variables from .env in the project root
from openai import OpenAI
client_openai = OpenAI()

# --- Load environment variables ---
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found in environment. Please add it to your .env file.")
    st.stop()

# Load ElevenLabs credentials
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    st.error("ElevenLabs API key not found in environment. Please add it to your .env file.")
    st.stop()

voice_id = os.getenv("ELEVENLABS_VOICE_ID")
if not voice_id:
    st.error("ElevenLabs voice ID not found in environment. Please add it to your .env file.")
    st.stop()

# --- Weaviate Client for Hybrid Search ---


def synthesize_speech_sdk(text):
    """
    Convert the provided text to speech using the ElevenLabs SDK.
    If the returned object is a generator, join the chunks into a bytes object.
    """
    client_eleven = ElevenLabs(api_key=elevenlabs_api_key)
    audio = client_eleven.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )
    if hasattr(audio, '__iter__') and not isinstance(audio, bytes):
        audio = b"".join(audio)
    return audio

def get_embedding(text, model="text-embedding-3-small"):
    """
    Get the embedding vector for the given text using OpenAI's embeddings API.
    """
    cleaned_text = text.replace("\n", " ")
    response = client_openai.embeddings.create(input=[cleaned_text], model=model)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)


# --- NEW: Function to generate a refined search query from the user prompt ---
def generate_search_query(user_prompt):
    search_prompt = f"Extract the key search terms from the following question to retrieve relevant transcript content:\n\n{user_prompt}\n\nSearch Query:"
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are an assistant that extracts concise keywords for search queries."},
            {"role": "user", "content": search_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# --- NEW: Hybrid search in Weaviate ---
def weaviate_hybrid_search(search_query, limit=3):
    # Query the "Transcript" collection in Weaviate using hybrid search.
    # Get the collection
    wv_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WCD_DEMO_URL"),  # Your Weaviate Cloud URL
    auth_credentials=Auth.api_key(os.getenv("WCD_DEMO_ADMIN_KEY")),  # Your admin key
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
    )
    Transcript = wv_client.collections.get("Transcript")

    # Perform query
    response = Transcript.query.hybrid(
        query=search_query, limit=limit, return_metadata=wq.MetadataQuery(score=True)
        )   
    wv_client.close()
    return response

# --- Sidebar Setup ---
with st.sidebar:
    st.markdown("**Leadership Coach Chatbot**")
    st.markdown("[View the source code](https://github.com/omerozerr/leadership-chatbot)")
    if st.button("Clear Conversation"):
        st.session_state["messages"] = [
            {"role": "system", "content": (
                "You are a knowledgeable Leadership Coach specializing in leadership "
                "strategies, professional development, and business acumen. Provide detailed "
                "and insightful responses tailored to leadership topics. "
                "The current year is 2025. When performing any actions (such as web searches), "
                "update all date references to 2025 and ignore any data labeled with earlier years."
                "When answering, please provide only your final answer in plain text. Do not output any chain-of-thought or internal reasoning"
            )},
            {"role": "assistant", "content": "Merhabalar, Ben liderlik koçunuz. Nasıl yardımcı olabilirim?"}
        ]
        st.rerun()

# --- Main Title and Description ---
st.title("Leadership Coach Chatbot")
st.write("This prototype integrates transcript-based context retrieval using Weaviate hybrid search, along with a chat interface using LangChain, Streamlit, and ElevenLabs for voice synthesis.")

# --- Initialize Conversation History if not present ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": (
             "You are a knowledgeable Leadership Coach specializing in leadership "
             "strategies, professional development, and business acumen. Provide detailed "
             "and insightful responses tailored to leadership topics. "
             "The current year is 2025. When performing any actions (such as web searches), "
             "update all date references to 2025 and ignore any data labeled with earlier years."
             "When answering, please provide only your final answer in plain text. Do not output any chain-of-thought or internal reasoning"
        )},
        {"role": "assistant", "content": "Merhabalar, Ben liderlik koçunuz. Nasıl yardımcı olabilirim?"}
    ]

# --- Display Conversation History (excluding system messages) ---
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

# --- Chat Input and Agent Response ---
if prompt := st.chat_input("Enter your question here..."):
    # Append user message to conversation history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # --- NEW STEP 1: Generate a refined search query from the user prompt ---
    search_query = generate_search_query(prompt)
    print("Generated search query:", search_query)
    
    # --- NEW STEP 2: Perform hybrid search in Weaviate using the search query ---
    hybrid_results = weaviate_hybrid_search(search_query, limit =4)
    # Determine the highest similarity score if available (Weaviate may return a score in metadata)
    # For simplicity, assume if any objects are returned, we use them.
    retrieved_context = ""
    reference_list = ""
    highest_score = 0
    if hybrid_results:
        # Build context: concatenate the "text" field from each object.
        for o in hybrid_results.objects:
            if o.metadata.score > highest_score:
                highest_score = o.metadata.score
            print("SKOR: ", o.metadata.score)
            retrieved_context += o.properties["text"] + "\n"
        print("-----------------------")
        print("Retrieved context:", retrieved_context)
        print("-----------------------")
        # Build reference list: for each object, create a reference with the video link.
        reference_list = "\n".join([f"Video: {obj.properties['video_url']}" for obj in hybrid_results.objects])

    # --- Define a threshold for using specialized context ---
    SIMILARITY_THRESHOLD = 0.75  # adjust as needed

    # Only use specialized context if found (you might also check for a minimum score if available)
    if retrieved_context and highest_score >= SIMILARITY_THRESHOLD:
        context_message = f"""do not web search if it is not necessary. Most likely, you already have the required information in the specialized transcripts below.
        Context from specialized transcripts (hybrid search results):
        {retrieved_context}

        References:
        {reference_list}

        You must use the above context to inform your answer, and include these references at the end of your response."""
        st.session_state["messages"].append({"role": "system", "content": context_message})
    
    # --- Continue with the normal agent call ---
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=openai.api_key,
        streaming=True
    )
    
    search_tool = DuckDuckGoSearchRun(name="Search")
    
    agent = initialize_agent(
        [search_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )
    
    callback_handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    
    response = agent.run(st.session_state["messages"], callbacks=[callback_handler])
    
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
    
    audio_bytes = synthesize_speech_sdk(response)
    st.session_state["audio_bytes"] = audio_bytes

# --- Voice Output Button ---
if st.session_state.get("audio_bytes"):
    if st.button("Play Voice Output"):
        play(st.session_state["audio_bytes"])
