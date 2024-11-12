import streamlit as st
import os
import time
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state for chat
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# Define available models
ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022"
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o1-mini",
    "o1-preview"
]

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize API clients
client = OpenAI(api_key=openai_api_key)
anthropic = Anthropic(api_key=anthropic_api_key)

def get_youtube_id(url):
    """Extract video ID from YouTube URL"""
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        return url.split("v=")[1].split("&")[0]
    return url

def get_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)
        return text_transcript
    except Exception as e:
        st.error(f"Error getting transcript: {str(e)}")
        return None

def chunk_text(text, chunk_size=2000):
    """Split text into chunks of approximately equal size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def summarize_openai(chunks, system_prompt):
    """Summarize text using OpenAI's API"""
    summaries = []
    for chunk in chunks:
        combined_prompt = f"{system_prompt}\n\nText to summarize: {chunk}"
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
        )
        summaries.append(response.choices[0].message.content)
    return summaries

def summarize_claude(chunks, system_prompt):
    """Summarize text using Claude's API"""
    summaries = []
    for chunk in chunks:
        response = anthropic.messages.create(
            model=st.session_state.model_name,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"{system_prompt}\n\nText to summarize: {chunk}"
            }]
        )
        summaries.append(response.content[0].text)
    return summaries

def chat_with_transcript(question, model_name):
    """Chat with the transcript content"""
    if not st.session_state.transcript_text:
        return "Please load a transcript first by entering a YouTube URL and clicking Summarize."
    
    system_prompt = """You are a helpful assistant analyzing a YouTube video transcript. 
    Answer questions about the content based on the transcript provided. 
    If the answer cannot be found in the transcript, say so clearly."""
    
    if model_name in ANTHROPIC_MODELS:
        response = anthropic.messages.create(
            model=model_name,
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"{system_prompt}\n\nTranscript: {st.session_state.transcript_text}\n\nQuestion: {question}"
            }]
        )
        return response.content[0].text
    else:
        combined_prompt = f"{system_prompt}\n\nTranscript: {st.session_state.transcript_text}\n\nQuestion: {question}"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
        )
        return response.choices[0].message.content

def summarize_youtube_video(youtube_url, progress_bar, progress_text, summarization_function):
    # Get video ID and transcript
    video_id = get_youtube_id(youtube_url)
    progress_text.text("Getting transcript...")
    transcript = get_transcript(video_id)
    
    if not transcript:
        return None, None
    
    # Store transcript in session state for chat functionality
    st.session_state.transcript_text = transcript
    
    progress_bar.progress(0.33)
    
    # Chunk the transcript
    progress_text.text("Processing transcript...")
    transcript_chunks = chunk_text(transcript)
    progress_bar.progress(0.66)
    
    # Generate summaries with timer
    start_time = time.time()
    system_prompt = "You are a helpful assistant that summarizes YouTube videos. Summarize the current chunk to succinct and clear bullet points of its contents."
    
    # Create placeholder for timer
    timer_placeholder = st.empty()
    
    def update_timer():
        elapsed_time = int(time.time() - start_time)
        timer_placeholder.text(f"Generating summary... (Time elapsed: {elapsed_time} seconds)")
    
    # Initial timer display
    update_timer()
    
    # Start summarization with periodic timer updates
    summaries = []
    for chunk in transcript_chunks:
        update_timer()
        chunk_summaries = summarization_function([chunk], system_prompt=system_prompt)
        summaries.extend(chunk_summaries)
    
    # Generate final TLDR
    update_timer()
    system_prompt_tldr = "You are a helpful assistant that summarizes YouTube videos. Someone has already summarized the video to key points. Summarize the key points to one or two sentences that capture the essence of the video."
    long_summary = "\n".join(summaries)
    short_summary = summarization_function([long_summary], system_prompt=system_prompt_tldr)[0]
    
    # Final timer update
    elapsed_time = int(time.time() - start_time)
    progress_text.text(f"Summary complete! (Total time: {elapsed_time} seconds)")
    progress_bar.progress(1.0)
    
    return long_summary, short_summary

# Streamlit interface
st.title("YouTube Video Summarizer & Chat")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Summarize", "Chat with Transcript"])

with tab1:
    youtube_url = st.text_input("Enter YouTube URL:")
    
    # Model selection with provider and specific model
    provider = st.radio("Select AI Provider:", ("OpenAI", "Anthropic"))
    
    if provider == "OpenAI":
        model_name = st.selectbox("Select OpenAI Model:", OPENAI_MODELS)
    else:
        model_name = st.selectbox("Select Anthropic Model:", ANTHROPIC_MODELS)
    
    st.session_state.model_name = model_name

    if st.button("Summarize"):
        if youtube_url:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Select summarization function based on user choice
            summarization_function = summarize_claude if provider == "Anthropic" else summarize_openai
            
            # Get summaries
            long_summary, short_summary = summarize_youtube_video(
                youtube_url, 
                progress_bar, 
                progress_text, 
                summarization_function
            )
            
            if long_summary and short_summary:
                st.subheader("TL;DR")
                st.write(short_summary)
                
                st.subheader("Detailed Summary")
                st.write(long_summary)
        else:
            st.error("Please enter a YouTube URL")

with tab2:
    if st.session_state.transcript_text is None:
        st.warning("Please load a transcript first by entering a YouTube URL in the Summarize tab.")
    else:
        # Chat interface
        st.subheader("Chat with the Transcript")
        
        # Add clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Display chat history
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.write(content)

        # Chat input
        if question := st.chat_input("Ask a question about the video content"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Get AI response
            with st.chat_message("user"):
                st.write(question)
                
            with st.chat_message("assistant"):
                response = chat_with_transcript(question, st.session_state.model_name)
                st.write(response)
                
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Sidebar instructions
st.sidebar.markdown("""
### How to use:
1. **Summarize Tab:**
   - Enter a YouTube URL
   - Select an AI model
   - Click 'Summarize'
   - Get both a short TL;DR and detailed summary

2. **Chat Tab:**
   - After loading a transcript, switch to the Chat tab
   - Ask questions about the video content
   - Get AI-powered responses based on the transcript
""")