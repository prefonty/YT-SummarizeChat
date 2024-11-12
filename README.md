# Summarize Youtube Videos and Chat with the summary

## Overview

### What does it do?

The YouGPTube Summarizer is a Python-based application that utilizes advanced language models from OpenAI and Anthropic to summarize YouTube videos. Given a YouTube URL, it fetches the video transcript and generates both a detailed summary and a TL;DR version. The summarization can be done using either OpenAI's models (including GPT-4) or Anthropic's Claude-3.5 models. Additionally, users can chat with the transcript content using the integrated chat interface.

### Why is it useful?

Ever felt overwhelmed by the amount of content in a lengthy YouTube video and wished for a concise summary? The YouGPTube Summarizer can help you get the essence of a video in a fraction of the time it takes to watch it. The interactive chat feature allows you to ask specific questions about the video content, making it perfect for research, study, or quick content review.

## Features

- Transcript-based summarization
- Dual summarization: detailed summary and TL;DR
- Interactive chat interface to ask questions about the video content
- Support for multiple AI models:
  - Anthropic: claude-3.5-sonnet, claude-3.5-haiku
  - OpenAI: gpt-4o, gpt-4o-mini, o1-mini, o1-preview
- Clean tabbed interface for different functionalities
- Real-time processing status and timing information

### Tech Implementation

Under the hood, the application uses several Python libraries:
- `streamlit` for the web interface
- `youtube_transcript_api` for transcript fetching
- `openai` and `anthropic` for AI processing
- `python-dotenv` for environment management

## Prerequisites

### Python Dependencies

All Python dependencies are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

### API Keys

You'll need to obtain API keys for OpenAI and Anthropic. Store these keys in `.env`:

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Running the Application

To run the app, navigate to the directory where the code is located and run:

```bash
streamlit run trans.py
```

## Usage Guide

1. **Summarize Tab:**
   - Enter a YouTube URL
   - Select your preferred AI provider (OpenAI or Anthropic)
   - Choose a specific model
   - Click 'Summarize' to generate both a TL;DR and detailed summary
   - Watch the real-time progress and timing information

2. **Chat Tab:**
   - After loading a transcript in the Summarize tab
   - Switch to the Chat tab
   - Ask questions about the video content
   - Get AI-powered responses based on the transcript
   - Use the 'Clear Chat History' button to start fresh

## Code Structure

### Key Functions

- `get_youtube_id(url)`: Extracts video ID from YouTube URL
- `get_transcript(video_id)`: Fetches video transcript using YouTube API
- `chunk_text(text)`: Splits text into manageable chunks
- `summarize_openai(chunks, system_prompt)`: Generates summary using OpenAI models
- `summarize_claude(chunks, system_prompt)`: Generates summary using Claude models
- `chat_with_transcript(question, model_name)`: Handles chat functionality
- `summarize_youtube_video()`: Main orchestration function

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

[Your chosen license]

## Acknowledgments

- OpenAI for their GPT models
- Anthropic for Claude-3.5
- YouTube for their transcript API
