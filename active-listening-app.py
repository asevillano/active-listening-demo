"""
Active Listening Demo Application

This application demonstrates a real-time active listening system that:

1. Captures audio from multiple sources:
   - System microphone (default)
   - Pre-recorded audio files (.wav)
   - Browser microphone (using WebRTC)

2. Processes audio using Azure AI services:
   - Azure AI Speech: Transcribes audio to text in real-time
   - Azure AI Language: Detects and masks personally identifiable information (PII)
   - Azure AI Agents: Generates intelligent responses based on context

3. Key features:
   - Interactive web interface built with Streamlit
   - Real-time processing with threading to avoid blocking the UI
   - Support for local Docker containers or cloud services
   - Automatic masking of sensitive data (names, addresses, phone numbers, etc.)
   - Visualization in separate columns: original text, masked text, and agent responses

4. Use cases:
   - Conversational AI demonstrations
   - Customer service systems with privacy protection
   - Real-time conversation analysis
   - Voice-enabled virtual assistants

Author: Angel Sevillano (Microsoft)
Version: 2025
"""

import re
import os, time, threading, queue, warnings  
from dotenv import load_dotenv  
  
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
from scipy import signal

# Suppress pygame warnings before importing
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"  # Hide pygame welcome message

import pygame
  
import azure.cognitiveservices.speech as speech_sdk  
from azure.core.credentials import AzureKeyCredential  
from azure.ai.textanalytics import TextAnalyticsClient, PiiEntityCategory  
from azure.ai.agents import AgentsClient  
from azure.ai.agents.models import ListSortOrder  
from azure.identity import DefaultAzureCredential

# Import pycaw for microphone control on Windows
try:
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False  

# ───────────────────────── Global variables for TTS ──────────────────────
# Simple global flag: True when TTS audio is playing
tts_playing = False
# Timestamp of the end of the last TTS (to ignore audio captured during TTS)
tts_end_time = 0.0
# Flag to track if pygame mixer is initialized (avoid re-initialization)
_pygame_mixer_initialized = False

# ───────────────────────── Microphone control (Windows) ───────────────────
# Cache for the microphone device (avoids repeated calls)
_microphone_volume_interface = None

def _get_microphone_volume():
    """Gets the microphone volume interface (with cache)"""
    global _microphone_volume_interface
    
    if not PYCAW_AVAILABLE:
        return None
    
    if _microphone_volume_interface is None:
        try:
            devices = AudioUtilities.GetMicrophone()
            if devices:
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                _microphone_volume_interface = interface.QueryInterface(IAudioEndpointVolume)
        except Exception as e:
            print(f"[DEBUG] Error getting microphone interface: {e}")
            return None
    
    return _microphone_volume_interface

def mute_microphone():
    """Mutes the system microphone (Windows only with pycaw)"""
    volume = _get_microphone_volume()
    if volume:
        try:
            volume.SetMute(1, None)
            print("[DEBUG] 🔇 Microphone MUTED")
            return True
        except Exception as e:
            print(f"[DEBUG] Error muting microphone: {e}")
    return False

def unmute_microphone():
    """Unmutes the system microphone (Windows only with pycaw)"""
    volume = _get_microphone_volume()
    if volume:
        try:
            volume.SetMute(0, None)
            print("[DEBUG] 🔊 Microphone UNMUTED")
            return True
        except Exception as e:
            print(f"[DEBUG] Error unmuting microphone: {e}")
    return False


def init_pygame_mixer():
    """Initialize pygame mixer once and keep it ready for fast playback"""
    global _pygame_mixer_initialized
    
    if not _pygame_mixer_initialized:
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            _pygame_mixer_initialized = True
            print("[DEBUG] pygame.mixer initialized successfully for TTS playback")
        except Exception as e:
            print(f"[DEBUG] Error initializing pygame mixer: {e}")
            _pygame_mixer_initialized = False
    
    return _pygame_mixer_initialized

# ───────────────────────── 1. Azure Services ───────────────────────────────  
PII_CATEGORIES = [  
    PiiEntityCategory.ADDRESS, PiiEntityCategory.CREDIT_CARD_NUMBER,  
    PiiEntityCategory.EMAIL,   PiiEntityCategory.ESDNI,  
    PiiEntityCategory.INTERNATIONAL_BANKING_ACCOUNT_NUMBER,  
    PiiEntityCategory.IP_ADDRESS, PiiEntityCategory.PERSON,  
    PiiEntityCategory.PHONE_NUMBER, PiiEntityCategory.SWIFT_CODE,  
]  
  
  
def init_services():  
    """Reads .env and builds Speech, Language, and Agents clients."""  
    load_dotenv(override=True)  
    
    # Show pycaw status once
    if PYCAW_AVAILABLE:
        print("[DEBUG] pycaw available - microphone muting enabled")
    else:
        print("[DEBUG] pycaw not available - using fallback timing method")
    
    # Read all environment variables once
    env_vars = {
        "SPEECH_KEY": os.getenv("SPEECH_KEY"),
        "SPEECH_REGION": os.getenv("SPEECH_REGION"),
        "STT_LOCAL_URL": os.getenv("STT_LOCAL_URL"),
        "TTS_LOCAL_URL": os.getenv("TTS_LOCAL_URL"),
        "SPEECH_LANGUAGE": os.getenv("SPEECH_LANGUAGE"),
        "DOCKER_STT": os.getenv("DOCKER_STT", "true"),
        "DOCKER_TTS": os.getenv("DOCKER_TTS", "true"),
        "AI_SERVICE_ENDPOINT": os.getenv("AI_SERVICE_ENDPOINT"),
        "AI_SERVICE_KEY": os.getenv("AI_SERVICE_KEY"),
        "AI_LOCAL_URL": os.getenv("AI_LOCAL_URL"),
        "TEXT_LANGUAGE": os.getenv("TEXT_LANGUAGE"),
        "DOCKER_AI": os.getenv("DOCKER_AI", "true"),
        "PROJECT_ENDPOINT": os.getenv("PROJECT_ENDPOINT"),
        "AGENT_ID": os.getenv("AGENT_ID"),
        "TTS_VOICE": os.getenv("TTS_VOICE", "en-US-JennyNeural"),
    }
    
    # Convert boolean flags once
    setup_vars = {
        **env_vars,
        "DOCKER_STT": env_vars["DOCKER_STT"].lower() == "true",
        "DOCKER_TTS": env_vars["DOCKER_TTS"].lower() == "true",
        "DOCKER_AI": env_vars["DOCKER_AI"].lower() == "true",
    }
    
    #missing_vars = [var for var, value in setup_vars.items() if not value]
    #if missing_vars:
    #    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
  
    # Setup AI Speech STT client
    print(f"[DEBUG] Initializing Azure Speech STT client...")  # DEBUG
    if setup_vars["DOCKER_STT"]: # Using Docker Speech container
        stt_cfg = speech_sdk.SpeechConfig(  
            host=setup_vars["STT_LOCAL_URL"],  
            speech_recognition_language=setup_vars["SPEECH_LANGUAGE"],  
        )  
        print(f"[DEBUG] Azure Speech client ready at {setup_vars['STT_LOCAL_URL']}...")  # DEBUG
    else: # Using Azure Speech Service in the cloud
        stt_cfg = speech_sdk.SpeechConfig(  
            subscription=setup_vars["SPEECH_KEY"],  
            region=setup_vars["SPEECH_REGION"],  
            speech_recognition_language=setup_vars["SPEECH_LANGUAGE"],  
        )  
        print(f"[DEBUG] Azure Speech STT client ready at {setup_vars['SPEECH_REGION']}...")  # DEBUG

    # Setup AI Speech TTS client
    print(f"[DEBUG] Initializing Azure Speech TTS client...")  # DEBUG
    if setup_vars["DOCKER_TTS"]: # Using Docker Speech container
        tts_cfg = speech_sdk.SpeechConfig(  
            host=setup_vars["TTS_LOCAL_URL"],  
        )  
        print(f"[DEBUG] Azure Speech TTS client ready at {setup_vars['TTS_LOCAL_URL']}...")  # DEBUG
    else: # Using Azure Speech Service in the cloud
        tts_cfg = speech_sdk.SpeechConfig(  
            subscription=setup_vars["SPEECH_KEY"],  
            region=setup_vars["SPEECH_REGION"],
        )  
        print(f"[DEBUG] Azure Speech TTS client ready at {setup_vars['SPEECH_REGION']}...")  # DEBUG
    # Configure the speech synthesizer
    tts_cfg.speech_synthesis_voice_name = setup_vars["TTS_VOICE"]
    # Use an in-memory synthesizer (no output file)
    synthesizer = speech_sdk.SpeechSynthesizer(speech_config=tts_cfg, audio_config=None)

    # Setup AI Language client
    print("[DEBUG] Initializing Azure Language client...")  # DEBUG
    if setup_vars["DOCKER_AI"]:  # Using Docker AI Language container
        ai_endpoint = setup_vars["AI_LOCAL_URL"]
    else:
        ai_endpoint = setup_vars["AI_SERVICE_ENDPOINT"]
    lang_cli = TextAnalyticsClient(  
        endpoint=ai_endpoint,  
        credential=AzureKeyCredential(setup_vars["AI_SERVICE_KEY"]),  
    )  
    print(f"[DEBUG] Azure Language client ready at {ai_endpoint}...")  # DEBUG

    # Setup AI Agents client
    print("[DEBUG] Initializing AI Agents client...")  # DEBUG
    ag_cli = AgentsClient(  
        endpoint=setup_vars["PROJECT_ENDPOINT"],  
        credential=DefaultAzureCredential(),  
    )  
    agent = ag_cli.get_agent(setup_vars['AGENT_ID'])  
    thread = ag_cli.threads.create()  
    print(f"[DEBUG] AI Agents client ready at {setup_vars['PROJECT_ENDPOINT']} with agent ID {setup_vars['AGENT_ID']}...")  # DEBUG

    return stt_cfg, tts_cfg, lang_cli, ag_cli, agent, thread, synthesizer, setup_vars["TTS_VOICE"]  
  
  
def mask_pii(lang_cli, text, language):  
    # Remove '.' from any text with the format nnn.nnn.nnn to avoid misdetection of phone numbers
    text = re.sub(r'(\d)\.(\d)', r'\1\2', text).strip()
    print(f"[DEBUG] Text after removing dots in numbers: '{text}'")
    
    res = lang_cli.recognize_pii_entities(  
        documents=[text],  
        language=language,  
        categories_filter=PII_CATEGORIES,  
    )[0]  
    return res.redacted_text if res.entities else text  
  
  
def ask_agent(ag_cli, agent_id, thread_id, content):  
    try:
        print(f"[DEBUG] Sending to agent: {content}")
        ag_cli.messages.create(thread_id=thread_id, role="user", content=content)  
        run = ag_cli.runs.create_and_process(thread_id=thread_id, agent_id=agent_id)  
        if run.status == "failed":  
            print(f"[DEBUG] Agent run failed!")
            return "⚠️ Agent run failed"  
      
        answer = ""  
        messages = ag_cli.messages.list(thread_id, order=ListSortOrder.ASCENDING)
        for m in messages:  
            if m.run_id == run.id and m.text_messages:  
                answer += m.text_messages[-1].text.value + "\n"
        print(f"[DEBUG] Agent answer: {answer}")
        return answer if answer else "No response from agent"
    except Exception as e:
        print(f"[DEBUG] Error in ask_agent: {e}")
        import traceback
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}"  


def should_discard_tts_echo(text):
    """
    Determines if the transcribed text should be discarded as TTS echo.
    Returns True if it should be discarded, False if valid.
    """
    global tts_playing, tts_end_time
    
    if not text:
        return True
    
    current_time = time.time()
    time_since_tts = current_time - tts_end_time if tts_end_time > 0 else 999
    
    # Log for debugging
    print(f"[DEBUG] Transcribed: '{text}' | tts_playing={tts_playing} | time_since_tts={time_since_tts:.2f}s")
    
    # If TTS is playing OR just finished, DISCARD
    # With pycaw, we use a shorter window (0.5s) since the microphone is muted
    time_since_tts_end = current_time - tts_end_time
    discard_window = 0.5 if PYCAW_AVAILABLE else 1.5
    
    if tts_playing or (tts_end_time > 0 and time_since_tts_end < discard_window):
        print(f"[DEBUG] ⚠️ DISCARDED (TTS active or recent: {time_since_tts_end:.2f}s ago) - '{text}'")
        return True
    
    return False


def process_transcribed_text(text, lang_cli, language, ag_cli, agent, thread, queue_out, 
                             enable_tts=False, synthesizer=None, tts_voice=None, stop_flag=None):
    """
    Processes transcribed text: masks PII, queries the agent, and optionally synthesizes response.
    This function centralizes the shared logic between speech_worker and browser_audio_worker.
    """
    try:
        # Check if it should be discarded due to TTS echo
        if should_discard_tts_echo(text):
            return
        
        # Mask PII
        print(f"[DEBUG] Calling mask_pii...")
        start_time = time.time()
        masked = mask_pii(lang_cli, text, language)
        end_time = time.time()
        print(f"[DEBUG] Masked text: '{masked}' completed in {end_time - start_time:.2f} seconds.")
        
        # Add transcription immediately (without response yet)
        data_partial = dict(original=text, masked=masked, response="⏳ Processing...")
        queue_out.put(("add", data_partial))
        
        # Query the agent
        print(f"[DEBUG] Calling ask_agent...")
        start_time = time.time()
        answer = ask_agent(ag_cli, agent.id, thread.id, masked)
        end_time = time.time()
        print(f"[DEBUG] Agent answer received: '{answer}' completed in {end_time - start_time:.2f} seconds.")
        
        # Update with the agent's response
        queue_out.put(("update", answer))
        
        # If TTS is enabled, synthesize the response
        if enable_tts and synthesizer and tts_voice and answer and not answer.startswith("⚠️"):
            tts_total_start = time.time()
            print(f"[DEBUG] Starting TTS synthesis and playback...")
            audio_data = synthesize_speech(synthesizer, answer, tts_voice)
            
            if audio_data:
                synthesis_time = time.time() - tts_total_start
                print(f"[DEBUG] TTS synthesis completed in {synthesis_time:.2f}s, starting playback SYNCHRONOUSLY...")
                playback_start = time.time()
                play_audio_bytes(audio_data, stop_flag)
                playback_time = time.time() - playback_start
                total_time = time.time() - tts_total_start
                print(f"[DEBUG] TTS playback completed in {playback_time:.2f}s (total TTS time: {total_time:.2f}s)")
                print(f"[DEBUG] Ready for new input")
            else:
                print("[DEBUG] TTS synthesis failed")
    
    except Exception as e:
        print(f"[DEBUG] Error in process_transcribed_text: {e}")
        import traceback
        traceback.print_exc()



def synthesize_speech(synthesizer, text, voice_name):
    """
    Synthesizes text to speech using Azure AI Speech.
    Returns the synthesized audio as bytes or None if error.
    """
    try:
        synthesis_start = time.time()
        print(f"[DEBUG] Starting speech synthesis with voice: {voice_name}")
        print(f"[DEBUG] Text to synthesize: '{text[:100]}...' ({len(text)} chars)")
               
        # Synthesize the text
        result = synthesizer.speak_text_async(text).get()
        synthesis_end = time.time()
        
        if result.reason == speech_sdk.ResultReason.SynthesizingAudioCompleted:
            print(f"[DEBUG] Speech synthesis completed in {synthesis_end - synthesis_start:.2f}s")
            print(f"[DEBUG] Audio data size: {len(result.audio_data)} bytes")
            return result.audio_data
        elif result.reason == speech_sdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"[DEBUG] Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speech_sdk.CancellationReason.Error:
                print(f"[DEBUG] Error details: {cancellation_details.error_details}")
            return None
        else:
            print(f"[DEBUG] Unexpected result reason: {result.reason}")
            return None
    except Exception as e:
        print(f"[DEBUG] Error in synthesize_speech: {e}")
        import traceback
        traceback.print_exc()
        return None


def play_audio_bytes(audio_data, stop_flag=None):
    """
    Plays audio from bytes using pygame in a COMPLETELY SYNCHRONOUS manner.
    This function BLOCKS until playback finishes or stop_flag is activated.
    Mutes the microphone during playback if pycaw is available.
    
    Args:
        audio_data: Audio bytes to play
        stop_flag: Optional threading.Event to stop playback
    """
    global tts_playing, tts_end_time
    
    if not audio_data:
        print(f"[DEBUG] No audio data to play")
        return
    
    try:
        import io
        audio_stream = io.BytesIO(audio_data)
        
        # Mark that TTS is playing BEFORE starting
        tts_playing = True
        tts_start_time = time.time()
        print(f"[DEBUG] *** TTS PLAYBACK STARTED *** tts_playing = True at {tts_start_time:.2f}")
        
        # MUTE the microphone before playing
        mic_muted = mute_microphone()
        
        # Initialize pygame mixer if not already initialized (one-time operation)
        if not init_pygame_mixer():
            print("[DEBUG] Failed to initialize pygame mixer")
            tts_playing = False
            if mic_muted:
                unmute_microphone()
            return
        
        # Load and play audio (mixer already initialized, so this is fast)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        
        # SYNCHRONOUS BLOCKING: wait until completely finished or stop_flag is activated
        while pygame.mixer.music.get_busy():
            # Check if it should stop
            if stop_flag and stop_flag.is_set():
                print("[DEBUG] *** TTS PLAYBACK STOPPED by user ***")
                pygame.mixer.music.stop()
                break
            time.sleep(0.05)  # Check every 50ms
        
        # DON'T quit pygame mixer - keep it initialized for next playback
        # pygame.mixer.quit() <- REMOVED to avoid re-initialization delay
        
        # UNMUTE the microphone after playing
        if mic_muted:
            time.sleep(0.1)  # Small pause before unmuting
            unmute_microphone()
        
        # Mark that TTS finished AFTER completion
        tts_playing = False
        tts_end_time = time.time()
        duration = tts_end_time - tts_start_time
        print(f"[DEBUG] *** TTS PLAYBACK FINISHED *** tts_playing = False at {tts_end_time:.2f} (duration: {duration:.2f}s)")
        
    except Exception as e:
        print(f"[DEBUG] Error playing audio: {e}")
        tts_playing = False  # Ensure it's deactivated even if there's an error
        tts_end_time = time.time()
        # Ensure the microphone is unmuted even if there's an error
        unmute_microphone()
        import traceback
        traceback.print_exc()
  


def play_audio_file(audio_file, stop_flag):
    """
    Plays an audio file using pygame.
    Can be stopped if stop_flag is activated.
    """
    try:
        if not os.path.exists(audio_file):
            print(f"[DEBUG] Audio file not found: {audio_file}")
            return
        
        # Initialize pygame mixer if not already initialized
        if not init_pygame_mixer():
            print("[DEBUG] Failed to initialize pygame mixer")
            return
            
        pygame.mixer.music.load(audio_file)
        print(f"[DEBUG] Starting audio playback: {audio_file}")
        pygame.mixer.music.play()
        
        # Wait while audio is playing or until it stops
        while pygame.mixer.music.get_busy() and not stop_flag.is_set():
            time.sleep(0.1)
        
        if stop_flag.is_set():
            pygame.mixer.music.stop()
            print("[DEBUG] Audio playback stopped by user")
        else:
            print("[DEBUG] Audio playback finished")
        
        # DON'T quit pygame mixer - keep it initialized
        # pygame.mixer.quit() <- REMOVED
    except pygame.error as e:
        print(f"[DEBUG] Pygame error playing audio: {e}")
    except Exception as e:
        print(f"[DEBUG] Error playing audio: {e}")
  
  
# ───────────────────────── 2. Recognition in thread ────────────────────────  
def speech_worker(queue_out: queue.Queue, stop_flag: threading.Event,  
                  stt_cfg, tts_cfg, lang_cli, ag_cli, agent, thread, language, synthesizer, audio_file=None, 
                  enable_tts=False, tts_voice=None):  
    """  
    Background thread: listens to system microphone or processes audio file,
    and puts results in 'queue_out'. Does NOT touch the Streamlit API.
    """  
    global tts_playing
    
    print("[DEBUG] Speech worker started")  # DEBUG
    
    # If using an audio file, start playback in a separate thread
    playback_thread = None
    if audio_file:
        print(f"[DEBUG] Using audio file: {audio_file}")  # DEBUG
        audio_cfg = speech_sdk.AudioConfig(filename=audio_file)
        # Start audio playback in separate thread
        playback_thread = threading.Thread(
            target=play_audio_file,
            args=(audio_file, stop_flag),
            daemon=True
        )
        playback_thread.start()
    else:
        print(f"[DEBUG] Using microphone")  # DEBUG
        audio_cfg = speech_sdk.AudioConfig(use_default_microphone=True)
    
    recognizer = speech_sdk.SpeechRecognizer(stt_cfg, audio_cfg)
    print(f"[DEBUG] SpeechRecognizer created successfully")  # DEBUG
  
    def recognizing_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
        """Callback for partial recognition (useful for debugging)"""
        if evt.result.text:
            print(".", end="", flush=True)  # Indicate activity
            #print(f"[DEBUG] Recognizing (partial): '{evt.result.text}'")
    
    def final_result(evt: speech_sdk.SpeechRecognitionEventArgs):
        text = evt.result.text.strip()
        # Only process if not using audio file (for files, TTS doesn't apply)
        should_enable_tts = enable_tts and not audio_file
        process_transcribed_text(
            text, lang_cli, language, ag_cli, agent, thread, queue_out,
            enable_tts=should_enable_tts, synthesizer=synthesizer, 
            tts_voice=tts_voice, stop_flag=stop_flag
        )
    
    def session_started_cb(evt):
        print("[DEBUG] *** Speech recognition session STARTED ***")
    
    def session_stopped_cb(evt):
        print("[DEBUG] *** Speech recognition session STOPPED ***")
    
    def canceled_cb(evt: speech_sdk.SpeechRecognitionCanceledEventArgs):
        print(f"[DEBUG] *** Recognition CANCELED: {evt.reason} ***")
        if evt.reason == speech_sdk.CancellationReason.Error:
            print(f"[DEBUG] Error details: {evt.error_details}")
  
    # Connect all events
    recognizer.recognizing.connect(recognizing_cb)
    recognizer.recognized.connect(final_result)
    recognizer.canceled.connect(canceled_cb)
    recognizer.session_started.connect(session_started_cb)
    recognizer.session_stopped.connect(session_stopped_cb)
    
    recognizer.start_continuous_recognition()  
    print("[DEBUG] Continuous recognition started")  # DEBUG
  
    # Wait until the 'stop_flag' signal is activated
    while not stop_flag.is_set():  
        time.sleep(0.2)  
  
    print("[DEBUG] Stopping recognition...")  # DEBUG
    try:
        recognizer.stop_continuous_recognition()
    except:
        pass
    
    # Wait for audio playback to finish if it exists
    if playback_thread and playback_thread.is_alive():
        print("[DEBUG] Waiting for audio playback to stop...")
        playback_thread.join(timeout=2)  # Wait maximum 2 seconds
    
    # Ensure the microphone is unmuted when stopping the worker
    unmute_microphone()
    
    print("[DEBUG] Speech worker stopped")  # DEBUG


def browser_audio_worker(queue_out: queue.Queue, stop_flag: threading.Event,
                        stt_cfg, lang_cli, ag_cli, agent, thread, language, synthesizer, tts_voice,
                        audio_queue: queue.Queue, enable_tts=False):
    """
    Background thread: processes browser audio in real-time using PushAudioInputStream.
    Receives audio chunks from audio_queue and sends them to the recognizer.
    """
    global tts_playing
    
    print("[DEBUG] Browser audio worker started")
    
    try:
        # Create a PushAudioInputStream with the correct format
        # Azure Speech SDK expects: 16kHz, 16-bit, mono PCM
        stream_format = speech_sdk.audio.AudioStreamFormat(
            samples_per_second=16000,  # 16kHz
            bits_per_sample=16,         # 16-bit
            channels=1                  # mono
        )
        push_stream = speech_sdk.audio.PushAudioInputStream(stream_format)
        audio_cfg = speech_sdk.AudioConfig(stream=push_stream)
        
        recognizer = speech_sdk.SpeechRecognizer(stt_cfg, audio_cfg)
        
        # Callbacks for debugging
        def recognizing_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
            """Partial recognition event (useful for debugging)"""
            if evt.result.text:
                print(f"[DEBUG] Recognizing (partial): '{evt.result.text}'")
        
        def recognized_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
            """Final recognition event"""
            text = evt.result.text.strip()
            
            # Check if there was an error
            if evt.result.reason == speech_sdk.ResultReason.NoMatch:
                print(f"[DEBUG] No match: {evt.result.no_match_details}")
                return
            
            # Process the text using the shared function
            process_transcribed_text(
                text, lang_cli, language, ag_cli, agent, thread, queue_out,
                enable_tts=enable_tts, synthesizer=synthesizer, 
                tts_voice=tts_voice, stop_flag=stop_flag
            )
        
        def canceled_cb(evt: speech_sdk.SpeechRecognitionCanceledEventArgs):
            """Cancellation or error event"""
            print(f"[DEBUG] Recognition canceled: {evt.reason}")
            if evt.reason == speech_sdk.CancellationReason.Error:
                print(f"[DEBUG] Error details: {evt.error_details}")
        
        def session_started_cb(evt):
            print("[DEBUG] Speech recognition session started")
        
        def session_stopped_cb(evt):
            print("[DEBUG] Speech recognition session stopped")
        
        # Connect all events
        recognizer.recognizing.connect(recognizing_cb)
        recognizer.recognized.connect(recognized_cb)
        recognizer.canceled.connect(canceled_cb)
        recognizer.session_started.connect(session_started_cb)
        recognizer.session_stopped.connect(session_stopped_cb)
        
        recognizer.start_continuous_recognition()
        print("[DEBUG] Browser audio continuous recognition started")
        
        # Process browser audio continuously
        bytes_received = 0
        first_chunk = True
        while not stop_flag.is_set():
            try:
                # Get audio chunk from queue with timeout
                audio_data = audio_queue.get(timeout=0.5)
                if audio_data is not None:
                    if first_chunk:
                        #print(f"[DEBUG] First audio chunk received: {len(audio_data)} bytes")
                        first_chunk = False
                    
                    # Send audio to recognizer
                    push_stream.write(audio_data)
                    bytes_received += len(audio_data)
                    #if bytes_received % 32000 < len(audio_data):  # Log every ~2 seconds of audio
                    #    print(f"[DEBUG] Total audio sent: {bytes_received} bytes (~{bytes_received/32000:.1f} seconds)")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DEBUG] Error processing audio chunk: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("[DEBUG] Stopping browser audio recognition...")
        try:
            push_stream.close()
        except Exception as e:
            print(f"[DEBUG] Error closing push stream: {e}")
        
        try:
            recognizer.stop_continuous_recognition()
        except Exception as e:
            print(f"[DEBUG] Error stopping recognition: {e}")
        
        # Empty the audio queue to avoid residual data
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Ensure the microphone is unmuted when stopping the worker
        unmute_microphone()
        
        print("[DEBUG] Browser audio worker stopped")
        
    except Exception as e:
        print(f"[DEBUG] Error in browser_audio_worker: {e}")
        import traceback
        traceback.print_exc()
        # Ensure the microphone is unmuted even if there's an error
        unmute_microphone()

  
  
# ───────────────────────── 3. Streamlit Interface ────────────────────────────  
def main():  
    # Configure the page to use full width
    st.set_page_config(layout="wide", page_title="Active Listening Demo", page_icon="🎤")
    st.title("🎤 Active Listening Demo with PII Masking")
    warnings.filterwarnings("ignore",  
                            message="Thread .*: missing ScriptRunContext",  
                            category=RuntimeWarning)  
  
    # First-time state initialization ---------------------------------  
    if "queue" not in st.session_state:  
        try:
            unmute_microphone()  # Ensure microphone is active on startup
            
            # Pre-initialize pygame mixer for faster TTS playback
            init_pygame_mixer()
            
            (stt_cfg, tts_cfg, lang_cli, ag_cli,  
             agent, thread, synthesizer, tts_voice) = init_services()  
        except ValueError as e:
            st.error(f"❌ Configuration Error: {str(e)}")
            st.info("💡 Please check your .env file and ensure all required environment variables are set.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error initializing Azure services: {str(e)}")
            st.info("💡 Please verify your credentials and network connection.")
            st.stop()
  
        st.session_state.queue       = queue.Queue()          # message queue
        st.session_state.records     = []                     # visible data
        st.session_state.stop_flag   = threading.Event()      # stop thread
        st.session_state.stt_cfg     = stt_cfg
        st.session_state.tts_cfg     = tts_cfg
        st.session_state.lang_cli    = lang_cli
        st.session_state.ag_cli      = ag_cli
        st.session_state.agent       = agent
        st.session_state.thread      = thread
        st.session_state.tts_voice   = tts_voice
        st.session_state.synthesizer = synthesizer
        st.session_state.running     = False
        st.session_state.audio_file  = None
        st.session_state.audio_mode  = "System Microphone"
        st.session_state.enable_tts  = False
    
    # Initialize audio_queue as global variable (not in session_state due to thread issues)
    if not hasattr(st.session_state, 'audio_queue_instance'):
        st.session_state.audio_queue_instance = queue.Queue()  # Queue for browser audio
    
    # Cache environment variables to avoid repeated reads
    if not hasattr(st.session_state, 'env_cache'):
        st.session_state.env_cache = {
            'DOCKER_STT': os.getenv("DOCKER_STT", "false").lower() == "true",
            'DOCKER_TTS': os.getenv("DOCKER_TTS", "false").lower() == "true",
            'DOCKER_AI': os.getenv("DOCKER_AI", "false").lower() == "true",
            'STT_LOCAL_URL': os.getenv("STT_LOCAL_URL", "Not configured"),
            'TTS_LOCAL_URL': os.getenv("TTS_LOCAL_URL", "Not configured"),
            'SPEECH_REGION': os.getenv("SPEECH_REGION", "Not configured"),
            'AI_LOCAL_URL': os.getenv("AI_LOCAL_URL", "Not configured"),
            'AI_SERVICE_ENDPOINT': os.getenv("AI_SERVICE_ENDPOINT", "Not configured"),
            'TEXT_LANGUAGE': os.getenv("TEXT_LANGUAGE", "en"),
            'BROWSER_MICRO': os.getenv("BROWSER_MICRO", "false").lower() == "true"
        }

    # Service connection information -------------------------------------
    st.sidebar.header("Service Connection Info")
    
    # Azure AI Speech STT information
    if st.session_state.env_cache['DOCKER_STT']:
        st.sidebar.text(f"🎤 AI Speech STT (Docker): {st.session_state.env_cache['STT_LOCAL_URL']}")
    else:
        st.sidebar.text(f"🎤 AI Speech STT (Cloud): Region: {st.session_state.env_cache['SPEECH_REGION']}")

    # Azure AI Speech TTS information
    if st.session_state.env_cache['DOCKER_TTS']:
        st.sidebar.text(f"🎤 AI Speech TTS (Docker): {st.session_state.env_cache['TTS_LOCAL_URL']}")
    else:
        st.sidebar.text(f"🎤 AI Speech TTS (Cloud): Region: {st.session_state.env_cache['SPEECH_REGION']}")

    # Azure AI Language information
    if st.session_state.env_cache['DOCKER_AI']:
        st.sidebar.text(f"🔤 AI Language (Docker): {st.session_state.env_cache['AI_LOCAL_URL']}")
    else:
        st.sidebar.text(f"🔤 AI Language (Cloud): {st.session_state.env_cache['AI_SERVICE_ENDPOINT']}")

    # Audio input options -------------------------------------------
    st.sidebar.header("Audio Input Settings")
    
    # Audio input mode selector
    if st.session_state.env_cache['BROWSER_MICRO']:
        input_options = ["System Microphone", "Audio File", "Browser Microphone"]
    else:
        input_options = ["System Microphone", "Audio File"]
    audio_mode = st.sidebar.radio(
        "Audio Input Source",
        options=input_options,
        index=0 if not hasattr(st.session_state, 'audio_mode') else input_options.index(st.session_state.audio_mode),
        disabled=st.session_state.running,
        help="System Microphone: Uses your computer's default microphone\nAudio File: Processes a pre-recorded audio file\nBrowser Microphone: Uses your browser's microphone with real-time processing"
    )
    
    st.session_state.audio_mode = audio_mode
    
    audio_file_path = None
    if audio_mode == "Audio File":
        # Look for audio files in the audio_files folder
        audio_folder = os.path.join(os.path.dirname(__file__), "audio_files")
        if os.path.exists(audio_folder):
            audio_files = [f for f in os.listdir(audio_folder) 
                          if f.lower().endswith(('.wav'))]
            
            if audio_files:
                selected_file = st.sidebar.selectbox(
                    "Select an audio file",
                    options=audio_files,
                    disabled=st.session_state.running
                )
                if selected_file:
                    audio_file_path = os.path.join(audio_folder, selected_file)
                    # Validate that the file exists and is accessible
                    if os.path.exists(audio_file_path) and os.access(audio_file_path, os.R_OK):
                        st.sidebar.success(f"File selected: {selected_file}")
                    else:
                        st.sidebar.error(f"Cannot access file: {selected_file}")
                        audio_file_path = None
            else:
                st.sidebar.warning("No audio files found in audio_files folder")
        else:
            st.sidebar.error("audio_files folder not found")
    
    st.session_state.audio_file = audio_file_path
    
    # Checkbox for Text-to-Speech (only if not using audio file)
    if audio_mode != "Audio File":
        st.sidebar.header("Text-to-Speech Settings")
        enable_tts = st.sidebar.checkbox(
            "Enable TTS for agent responses",
            value=st.session_state.enable_tts,
            disabled=st.session_state.running,
            help=f"Synthesize agent responses using voice: {st.session_state.tts_voice}"
        )
        st.session_state.enable_tts = enable_tts
        
        if enable_tts:
            st.sidebar.info(f"🔊 Voice: {st.session_state.tts_voice}")
    else:
        # Disable TTS when using a file
        st.session_state.enable_tts = False
    
    # Sidebar status -------------------------------------------
    st.sidebar.header("Status")
    if audio_mode == "Browser Microphone":
        if st.session_state.running:
            st.sidebar.success("🌐 Browser microphone active")
        else:
            st.sidebar.info("🔈 Idle - Click START below")
    elif st.session_state.running:
        if audio_mode == "Audio File":
            st.sidebar.success("📁 Processing file...")
        else:
            st.sidebar.success("🎤 Listening from system...")
    else:
        st.sidebar.info("🔈 Idle")
  
    # If in browser mode, show WebRTC component first
    webrtc_ctx = None
    if audio_mode == "Browser Microphone":
        # Only show informative message if not active
        if not st.session_state.get("browser_running", False):
            st.info("🎙️ **Browser Microphone Mode** - Click START below to begin audio capture")
        
        # Get reference to the queue outside the callback (to avoid session_state issues in threads)
        audio_queue_ref = st.session_state.audio_queue_instance
        
        # Local variable to track if first frame was logged (don't use session_state in callbacks)
        audio_callback_logged = {'logged': False}
        
        def audio_frame_callback(frame: av.AudioFrame):
            """Callback that receives audio frames from the browser"""
            try:
                # Obtener la tasa de muestreo original
                original_sample_rate = frame.sample_rate
                target_sample_rate = 16000
                
                # Log del primer frame para debugging (solo una vez)
                if not audio_callback_logged['logged']:
                    #print(f"[DEBUG] First audio frame: rate={original_sample_rate}Hz, format={frame.format.name}, channels={frame.layout.name}")
                    audio_callback_logged['logged'] = True
                
                # Convertir el frame a numpy array
                audio_array = frame.to_ndarray()
                
                # Verificar el formato del audio
                # Si viene en int16 (s16), convertir a float primero para el resample
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32767.0
                
                # Handle multiple channels (stereo -> mono)
                if len(audio_array.shape) > 1:
                    # If stereo (C, N) where C is number of channels
                    if audio_array.shape[0] > 1:
                        # Average channels to convert to mono
                        audio_array = audio_array.mean(axis=0)
                    else:
                        # If shape is (1, N), flatten
                        audio_array = audio_array.flatten()
                
                # Resample if sample rate is different from 16kHz
                if original_sample_rate != target_sample_rate:
                    # Calculate number of samples after resampling
                    num_samples = int(len(audio_array) * target_sample_rate / original_sample_rate)
                    # Check that we have enough samples for resampling
                    if len(audio_array) > 0:
                        audio_array = signal.resample(audio_array, num_samples)
                    else:
                        print("[DEBUG] Warning: Empty audio array for resampling")
                        return frame
                
                # Convert from float [-1.0, 1.0] to int16 [-32768, 32767]
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    audio_array = (audio_array * 32767).astype(np.int16)
                elif audio_array.dtype != np.int16:
                    audio_array = audio_array.astype(np.int16)
                
                # Convert to bytes
                audio_data = audio_array.tobytes()
                
                # Send to queue for processing
                audio_queue_ref.put(audio_data)
            except Exception as e:
                print(f"[DEBUG] Error in audio_frame_callback: {e}")
                import traceback
                traceback.print_exc()
            return frame
        
        # Configure WebRTC for audio capture (simplified for faster loading)
        rtc_configuration = RTCConfiguration({"iceServers": []})  # No STUN for faster loading
        
        # Configure minimal audio constraints
        media_stream_constraints = {
            "video": False,
            "audio": True,  # Simplified
        }
        
        webrtc_ctx = webrtc_streamer(
            key="browser-audio-capture",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_configuration,
            media_stream_constraints=media_stream_constraints,
            audio_frame_callback=audio_frame_callback,
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            st.success("🎤 Microphone is active and capturing audio")
        elif webrtc_ctx.state.signalling:
            st.warning("⏳ Connecting to microphone...")
    
    # Update running state based on WebRTC context
    if audio_mode == "Browser Microphone" and webrtc_ctx:
        # In browser mode, "running" state is controlled by WebRTC
        browser_was_running = st.session_state.get("browser_running", False)
        browser_is_running = webrtc_ctx.state.playing
        
        # If WebRTC just started playing, start the worker
        if browser_is_running and not browser_was_running:
            # Verify that there's no worker already running
            if not st.session_state.running:
                st.session_state.stop_flag.clear()
                st.session_state.worker = threading.Thread(
                    target=browser_audio_worker,
                    args=(st.session_state.queue,
                          st.session_state.stop_flag,
                          st.session_state.stt_cfg,
                          st.session_state.lang_cli,
                          st.session_state.ag_cli,
                          st.session_state.agent,
                          st.session_state.thread,
                          st.session_state.env_cache['TEXT_LANGUAGE'],
                          st.session_state.synthesizer,
                          st.session_state.tts_voice,
                          st.session_state.audio_queue_instance,
                          st.session_state.enable_tts),
                    daemon=True,
                )
                st.session_state.worker.start()
                st.session_state.running = True
                print("[DEBUG] Browser audio worker started from UI")
        
        # If WebRTC stopped, stop the worker
        elif not browser_is_running and browser_was_running:
            st.session_state.stop_flag.set()
            st.session_state.running = False
            print("[DEBUG] Browser audio worker stopped from UI")
        
        st.session_state.browser_running = browser_is_running
  
    # Start / Stop button (only for non-browser modes) -----------------------
    if audio_mode != "Browser Microphone":
        col_btn, col_status = st.columns([1, 3])  
        if not st.session_state.running:  
            # Validate that if using file, one has been selected
            can_start = True
            if audio_mode == "Audio File" and not audio_file_path:
                can_start = False
                
            if col_btn.button("▶ Start", type="primary", disabled=not can_start):  
                st.session_state.stop_flag.clear()
                
                # System or file mode: use speech_worker
                st.session_state.worker = threading.Thread(  
                    target=speech_worker,  
                    args=(st.session_state.queue,  
                          st.session_state.stop_flag,  
                          st.session_state.stt_cfg,
                          st.session_state.tts_cfg,
                          st.session_state.lang_cli,
                          st.session_state.ag_cli,
                          st.session_state.agent,
                          st.session_state.thread,  
                          st.session_state.env_cache['TEXT_LANGUAGE'],
                          st.session_state.synthesizer,
                          audio_file_path if audio_mode == "Audio File" else None,
                          st.session_state.enable_tts,
                          st.session_state.tts_voice),
                    daemon=True,  
                )
                    
                st.session_state.worker.start()  
                st.session_state.running = True  
        else:  
            if col_btn.button("⏹ Stop", type="secondary"):  
                st.session_state.stop_flag.set()  
                st.session_state.running = False
                # Force immediate UI refresh to show Start button
                st.rerun()
    
    # Empty the results queue coming from threads -----------------------  
    new_items = 0
    while not st.session_state.queue.empty():  
        message = st.session_state.queue.get()
        
        if isinstance(message, tuple):
            action, data = message
            if action == "add":
                # Add new record with transcription (without complete response yet)
                st.session_state.records.append(data)
                new_items += 1
                #print(f"[DEBUG] Added partial record to queue: {data}")  # DEBUG
            elif action == "update":
                # Update last record with agent response
                if st.session_state.records:
                    st.session_state.records[-1]["response"] = data
                    #print(f"[DEBUG] Updated last record with agent response")  # DEBUG
        else:
            # Compatibility with old format (just in case)
            st.session_state.records.append(message)
            new_items += 1
            #print(f"[DEBUG] Added complete record: {message}")  # DEBUG
    
    #if new_items > 0:
    #    print(f"[DEBUG] Added {new_items} new items. Total records: {len(st.session_state.records)}")  # DEBUG

    #print(f"[DEBUG] Current records count: {len(st.session_state.records)}")  # DEBUG
  
    # Display results ------------------------------------------------------  
    left, right = st.columns(2)  
  
    #print(f"[DEBUG] Rendering UI with {len(st.session_state.records)} records")  # DEBUG
    
    with left:  
        st.subheader("Original transcription")  
        # Scrollable container to show original transcriptions
        with st.container(height=300):
            if st.session_state.records:
                #print(f"[DEBUG] Displaying {len(st.session_state.records)} original texts")  # DEBUG
                # Reverse order to show most recent first
                for i, rec in enumerate(reversed(st.session_state.records)):  
                    actual_index = len(st.session_state.records) - i
                    st.markdown(f"**{actual_index}.** {rec['original']}")  
                    st.divider()
            else:
                st.info("Waiting for speech input...")
  
        st.subheader("Masked transcription")  
        # Scrollable container to show masked transcriptions
        with st.container(height=300):
            if st.session_state.records:
                # Reverse order to show most recent first
                for i, rec in enumerate(reversed(st.session_state.records)):  
                    actual_index = len(st.session_state.records) - i
                    st.markdown(f"**{actual_index}.** {rec['masked']}")  
                    st.divider()
            else:
                st.info("Waiting for speech input...")
  
    with right:  
        st.subheader("Agent response")  
        # Scrollable container to show agent responses
        with st.container(height=620):
            if st.session_state.records:
                # Reverse order to show most recent first
                for i, rec in enumerate(reversed(st.session_state.records)):  
                    actual_index = len(st.session_state.records) - i
                    st.markdown(f"**{actual_index}.** {rec['response']}")  
                    st.divider()
            else:
                st.info("Waiting for speech input...")

    # Auto-refresh while recording
    if st.session_state.running:  
        time.sleep(1)  # 1 second pause to avoid saturation
        st.rerun() 
  
# ─────────────────────────────────────────────────────────────────────────────  
if __name__ == "__main__":  
    main()  