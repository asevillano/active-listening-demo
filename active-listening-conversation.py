"""
Active Listening Conversation Processor

This application processes pre-recorded conversation files to demonstrate Azure AI capabilities:

1. Audio File Processing:
   - Processes WAV audio files from the audio_files directory
   - Plays the audio file while processing for demonstration purposes
   - Supports various conversation types (customer service, appointment setting, order taking)

2. Azure AI Services Integration:
   - Azure AI Speech: Transcribes audio conversations to text
   - Azure AI Language: Detects and masks personally identifiable information (PII)
   - Azure AI Agents: Analyzes conversation content and provides intelligent insights

3. Key Features:
   - Batch processing of conversation recordings
   - Real-time audio playback during processing
   - Configurable for Docker containers or cloud services
   - PII protection and data privacy compliance
   - Structured output with analysis results

4. Use Cases:
   - Call center conversation analysis
   - Customer service quality assessment
   - Training data processing for AI models
   - Compliance and privacy auditing
   - Conversation insights and analytics

5. Sample Files Supported:
   - Live answering service calls
   - Customer support interactions
   - Product refund conversations
   - General call center recordings

Author: Angel Sevillano (Microsoft)
Version: 2025
"""

# Import packages
from dotenv import load_dotenv
from functions import *
import pygame

def play_audio(file_path):
    # Inicializar el mixer de pygame
    pygame.mixer.init()
    # Cargar el archivo WAV
    pygame.mixer.music.load(file_path)
    # Reproducir el audio
    pygame.mixer.music.play()

def main():

    try:
        # Get Configuration Settings
        load_dotenv(override=True)
        DOCKER_SPEECH, DOCKER_AI, SPEECH_LANGUAGE, TEXT_LANGUAGE = init_config()

        # Configure AI Speech Service
        speech_config = init_speech(DOCKER_SPEECH, SPEECH_LANGUAGE)

        # Configure AI Language Service
        ai_client = init_language(DOCKER_AI)

        # AI Projects Service
        agents_client, agent, thread = init_agents()

        # Process conversation from file
        FILE_NAME = "audio_files/Live Answering Service Sample Call - Appointment Setting.wav"
        #FILE_NAME = "audio_files/Sample Order Taking Customer Support Philippines.wav"
        #FILE_NAME = "audio_files/Customer Service Sample Call - Product Refund.wav"
        #FILE_NAME = "audio_files/callCenterRecording.wav" #"audio_files/AUD-20250902-WA0001.wav" #"audio_files/katiesteve.wav"

        # Play audio file        
        play_audio(FILE_NAME)
        
        print(f"Processing file: {FILE_NAME}\n")
        results = recognize_from_file(FILE_NAME, speech_config, ai_client, agents_client, agent, thread, TEXT_LANGUAGE)
        print("---------------------------------")
        print("FINAL RESULTS:\n")
        for r in results:
            print(r)

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()