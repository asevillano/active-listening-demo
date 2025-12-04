
"""
Active Listening Functions Library

This module provides core functions for Azure AI-powered speech processing and conversation analysis:

1. Service Initialization Functions:
   - init_config(): Loads environment configuration for Docker/cloud deployment
   - init_speech(): Configures Azure Speech Service for speech recognition
   - init_language(): Sets up Azure Language Service for text analytics
   - init_agents(): Initializes Azure AI Agents for intelligent responses

2. Speech Recognition Functions:
   - speech_recognize_continuous(): Real-time continuous speech recognition from microphone
   - recognize_from_file(): Batch processing of audio files with speaker identification
   - conversation_transcriber functions: Handle conversation transcription events

3. Text Processing Functions:
   - mask_pii_entities(): Detects and masks personally identifiable information (PII)
   - process_transcription(): Complete pipeline for speech-to-text-to-agent processing
   - send_and_run_message(): Sends queries to AI agents and retrieves responses

4. Event Callback Functions:
   - Session management callbacks (started, stopped, canceled)
   - Real-time transcription event handlers
   - Error handling and logging functions

5. Key Features:
   - Multi-language support for speech and text processing
   - PII protection with configurable entity categories
   - Speaker identification in conversation scenarios
   - Flexible deployment (Docker containers or Azure cloud)
   - Real-time and batch processing modes
   - Comprehensive error handling and logging

6. Configuration Options:
   - DOCKER_SPEECH/DOCKER_AI: Toggle between local containers and cloud services
   - SPEECH_LANGUAGE/TEXT_LANGUAGE: Configurable language settings
   - PII_CATEGORIES: Customizable list of entities to detect and mask

Author: Angel Sevillano (Microsoft)
Version: 2025
"""

# Import packages
import re
import os, time
import azure.cognitiveservices.speech as speech_sdk
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, PiiEntityCategory
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder, FilePurpose

# Constants / Configuration
PII_CATEGORIES = [PiiEntityCategory.ADDRESS,
                  PiiEntityCategory.CREDIT_CARD_NUMBER,
                  PiiEntityCategory.EMAIL,
                  PiiEntityCategory.ESDNI,
                  PiiEntityCategory.INTERNATIONAL_BANKING_ACCOUNT_NUMBER,
                  PiiEntityCategory.IP_ADDRESS,
                  PiiEntityCategory.PERSON,
                  PiiEntityCategory.PHONE_NUMBER,
                  PiiEntityCategory.SWIFT_CODE]

# Configure solution
def init_config():
    DOCKER_SPEECH   = os.getenv("DOCKER_STT", "true").lower() == "true"
    DOCKER_AI       = os.getenv("DOCKER_AI", "true").lower() == "true"
    SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE")
    TEXT_LANGUAGE   = os.getenv("TEXT_LANGUAGE")
    print(f"Configuration - DOCKER_SPEECH: {DOCKER_SPEECH}, DOCKER_AI: {DOCKER_AI}, SPEECH_LANGUAGE: {SPEECH_LANGUAGE}, TEXT_LANGUAGE: {TEXT_LANGUAGE}")

    return DOCKER_SPEECH, DOCKER_AI, SPEECH_LANGUAGE, TEXT_LANGUAGE


# Configure AI Speech Service
def init_speech(docker_speech, speech_language):
    if docker_speech: # Using local container
        speech_local_url = os.getenv('SPEECH_LOCAL_URL')
        if speech_language:
            print(f"Using speech language: {speech_language}")
            speech_config = speech_sdk.SpeechConfig(host=speech_local_url, speech_recognition_language=speech_language)
        else:
            speech_config = speech_sdk.SpeechConfig(host=speech_local_url)
        print('Ready to use speech service in:', speech_local_url)
    else:
        speech_key = os.getenv('SPEECH_KEY')
        speech_region = os.getenv('SPEECH_REGION')
        if speech_language:
            speech_config = speech_sdk.SpeechConfig(subscription=speech_key, region=speech_region, speech_recognition_language=speech_language)
        else:
            speech_config = speech_sdk.SpeechConfig(subscription=speech_key, region=speech_region)
        print('Ready to use speech service in:', speech_config.region) #os.getenv('SPEECH_LOCAL_URL')) 

    return speech_config

# Configure AI Language Service
def init_language(docker_ai):
    if docker_ai: # Using local container
        ai_endpoint = os.getenv('AI_LOCAL_URL')
    else:
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
    ai_key = os.getenv('AI_SERVICE_KEY')

    # Create client using endpoint and key
    ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))
    print('Ready to use AI Language service in:', ai_endpoint)

    return ai_client

# Configure AI Agents Service
def init_agents():
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    agent_id = os.getenv("AGENT_ID")
    agents_client = AgentsClient(endpoint=project_endpoint, credential=DefaultAzureCredential())

    # Get the agent client by its ID
    agent = agents_client.get_agent(agent_id)
    # Create a thread for communication
    thread = agents_client.threads.create()
    print(f"Created thread, ID: {thread.id}\n")

    return agents_client, agent, thread

# Function to create a thread and send a message to the agent
def send_and_run_message(agents_client, agent_id, thread_id, content):
    # Create message to thread
    message = agents_client.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )
    #print(f"Created message, ID: {message.id} with content: {content}")
    #try:
    run = agents_client.runs.create_and_process(thread_id=thread_id, agent_id=agent_id)
    #print(f"Run finished with status: {run.status}")
    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
        return

    # Get the last message from the sender
    messages = agents_client.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING)
    response=""
    for i, message in enumerate(messages):
        if message.run_id == run.id and message.text_messages:
            #print(f"[{i}]: {message.role}: {message.text_messages[-1].text.value}")
            #print(message.text_messages[-1].text.value)
            response+=message.text_messages[-1].text.value+"\n"
    return response #run
    
    #except Exception as e:
    #    print(f"Error: {e}")
    #    return None

# Process PII recognition and mask entities in the text
def mask_pii_entities(ai_client, text, text_language):
    # Remove '.' from any text with the format nnn.nnn.nnn to avoid misdetection of phone numbers
    text = re.sub(r'(\d)\.(\d)', r'\1\2', text).strip()
    print(f"[DEBUG] Text after removing dots in numbers: '{text}'")
    
    print("Processing PII recognition...")
    start_time = time.time()
    if text_language:
        print(f"Using text language: {text_language}")
        pii_entities = ai_client.recognize_pii_entities(documents=[text], language=text_language, categories_filter=PII_CATEGORIES)[0]
    else:
        pii_entities = ai_client.recognize_pii_entities(documents=[text], categories_filter=PII_CATEGORIES)[0]
    end_time = time.time()
    print(f"PII recognition completed in {end_time - start_time:.2f} seconds.")
    # Print recognized PII entities and redacted text
    if len(pii_entities.entities) > 0:
        print("\nPII Entities")
        for pii_entity in pii_entities.entities:
            print(f'\t{pii_entity.text} ({pii_entity.category})')
        print(f'Masked Text: {pii_entities.redacted_text}')
        masked_text = pii_entities.redacted_text
    else:
        masked_text = text

    return masked_text

# Callback functions

def process_transcription(evt: speech_sdk.SpeechRecognitionEventArgs, ai_client, agents_client, agent, thread, text_language, conversation_analysis=False):
    transcription=evt.result.text
    print(f'\nRECOGNIZED: {transcription}')
    
    # Process PII recognition
    masked_transcription = mask_pii_entities(ai_client, transcription, text_language)

    if conversation_analysis:
        user_query = f"Speaker: {evt.result.speaker_id}, Transcription: {masked_transcription}"
    else:
        user_query = masked_transcription

    # Send message to agent
    print(f'\nSending to agent: "{user_query}"')
    start_time = time.time()
    response = send_and_run_message(agents_client, agent.id, thread.id, user_query)
    end_time = time.time()
    print(f"Agent response received in {end_time - start_time:.2f} seconds.\n")
    print("##########################################################################")
    print(f"RESPONSE:\n{response}")
    print("##########################################################################")
    print('\nListening...')

def session_started_cb(evt: speech_sdk.SessionEventArgs):
    print('Listening...')

def session_canceled_cb(evt: speech_sdk.SessionEventArgs):
    print('Canceled event')

def session_stopped_cb(evt: speech_sdk.SessionEventArgs):
    print('Session Stopped event')

# Conduct continuous speech recognition from microphone
def speech_recognize_continuous(speech_config, ai_client, agents_client, agent, thread, text_language):
    import time
    
    # Performs continuous speech recognition from microphone
    audio_config = speech_sdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)
    transcribing_stop = False

    def stop_cb(evt: speech_sdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print(f'CLOSING on {evt}')
        nonlocal transcribing_stop
        transcribing_stop = True

    # Connect callbacks to the events fired by the speech recognizer
    # speech_recognizer.recognizing.connect(lambda evt: print(f'RECOGNIZING: {evt.result.text}'))
    #speech_recognizer.recognized.connect(lambda evt: print(f'RECOGNIZED: {evt.result.text}'))
    
    speech_recognizer.recognized.connect(lambda evt: process_transcription(evt, ai_client, agents_client, agent, thread, text_language, conversation_analysis = False))
    
    all_results = []
    def handle_final_result(evt):
        all_results.append(evt.result.text)
    speech_recognizer.recognized.connect(handle_final_result)
    speech_recognizer.session_started.connect(session_started_cb)
    speech_recognizer.session_stopped.connect(session_stopped_cb)
    speech_recognizer.canceled.connect(session_canceled_cb)
    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not transcribing_stop:
        time.sleep(.5)
    speech_recognizer.stop_continuous_recognition()

    return all_results


# Conversation transcriber from file
SHOW_TRANSCRIBED = True
SHOW_TRANSCRIBING = False

def conversation_transcriber_transcribed_cb(evt: speech_sdk.SpeechRecognitionEventArgs, ai_client, agents_client, agent, thread, text_language):
    transcription=evt.result.text
    if SHOW_TRANSCRIBED:
        if evt.result.reason == speech_sdk.ResultReason.RecognizedSpeech:
            print(f"\nTRANSCRIBED: {transcription}")
            print(f"Speaker ID: {evt.result.speaker_id}")
        elif evt.result.reason == speech_sdk.ResultReason.NoMatch:
            print(f'\tNOMATCH: Speech could not be TRANSCRIBED: {evt.result.no_match_details}')
    else:
        if evt.result.reason == speech_sdk.ResultReason.RecognizedSpeech:
            print('.', end='', flush=True)

    if transcription.strip() != "":
        process_transcription(evt, ai_client, agents_client, agent, thread, text_language, conversation_analysis=True)

def conversation_transcriber_transcribing_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
    if SHOW_TRANSCRIBING:
        print('TRANSCRIBING:')
        print(f'Text: {evt.result.text}')
        print(f'Speaker ID: {evt.result.speaker_id}')
    else:
        print('.', end='', flush=True)

def recognize_from_file(audio_filename, speech_config, ai_client, agents_client, agent, thread, text_language):
    # Verificar que el archivo existe
    if not os.path.exists(audio_filename):
        raise FileNotFoundError(f"No se encontró el archivo: {audio_filename}")
    
    audio_config = speech_sdk.audio.AudioConfig(filename=audio_filename)
    conversation_transcriber = speech_sdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)

    transcribing_stop = False

    def stop_cb(evt: speech_sdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print(f'CLOSING on {evt}')
        nonlocal transcribing_stop
        transcribing_stop = True

    # Connect callbacks to the events fired by the conversation transcriber
    conversation_transcriber.transcribed.connect(lambda evt: conversation_transcriber_transcribed_cb(evt, ai_client, agents_client, agent, thread, text_language))

    all_results = []
    def handle_final_result(evt):
        if evt.result.text:
            line = f"Speaker ID: {evt.result.speaker_id}\nText: {evt.result.text}\n"
            all_results.append(line)
    conversation_transcriber.transcribed.connect(handle_final_result)

    conversation_transcriber.transcribing.connect(conversation_transcriber_transcribing_cb)
    conversation_transcriber.session_started.connect(session_started_cb)
    conversation_transcriber.session_stopped.connect(session_stopped_cb)
    conversation_transcriber.canceled.connect(session_canceled_cb)
    # stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    conversation_transcriber.start_transcribing_async()

    # Waits for completion.
    while not transcribing_stop:
        time.sleep(.5)

    conversation_transcriber.stop_transcribing_async()

    return all_results