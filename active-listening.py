
"""
Active Listening Console Application

This is a command-line application that demonstrates real-time speech recognition and processing:

1. Real-time Speech Processing:
   - Continuous speech recognition from system microphone
   - Live transcription with immediate text output
   - Console-based interface for simple demonstrations

2. Azure AI Services Integration:
   - Azure AI Speech: Real-time speech-to-text conversion
   - Azure AI Language: PII detection and masking capabilities
   - Azure AI Agents: Intelligent conversation analysis and responses

3. Key Features:
   - Command-line simplicity for quick testing
   - Continuous listening mode until manually stopped
   - Configurable language support (English, Spanish, Turkish, etc.)
   - Support for both Docker containers and cloud services
   - Real-time PII protection and data masking

4. Use Cases:
   - Quick speech recognition testing
   - Command-line demonstrations
   - Development and debugging of speech workflows
   - Simple voice-to-text conversion
   - Proof-of-concept implementations

5. Configuration:
   - Multi-language support through environment variables
   - Flexible deployment options (local Docker or Azure cloud)
   - Easy switching between different AI service endpoints

Author: Angel Sevillano (Microsoft)
Version: 2025
"""

# Import packages
from dotenv import load_dotenv
from functions import *

'''
# Constants / Configuration
DOCKER_SPEECH=False
DOCKER_AI=False
SPEECH_LANGUAGE = 'en-US' #'tr-TR' #'es-es'
TEXT_LANGUAGE = 'en' #'tr'
'''

def main():

    #try:
    # Get Configuration Settings
    load_dotenv(override=True)
    DOCKER_SPEECH, DOCKER_AI, SPEECH_LANGUAGE, TEXT_LANGUAGE = init_config()

    # Configure AI Speech Service
    speech_config = init_speech(DOCKER_SPEECH, SPEECH_LANGUAGE)

    # Configure AI Language Service
    ai_client = init_language(DOCKER_AI)

    # AI Projects Service
    agents_client, agent, thread = init_agents()

    # Get spoken input
    transcription = speech_recognize_continuous(speech_config, ai_client, agents_client, agent, thread, TEXT_LANGUAGE)
    print(f"TRANSCRIPCIÓN COMPLETA:\n{transcription}")

    #except Exception as ex:
    #    print(ex)



if __name__ == "__main__":
    main()