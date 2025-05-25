import streamlit as st
import requests
import tempfile
import os
import numpy as np
import soundfile as sf
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import time
import queue

# Page setup
st.set_page_config(page_title="Grammar Analysis", layout="wide")
st.title("🎤 Speech to Grammar Analysis")
st.markdown("""
Speak into your microphone and get instant grammar feedback with:
- 🎙️ Speech-to-text transcription
- ✍️ Grammar correction
- 🔍 Error analysis
- 📊 Accuracy score
- 💡 Improvement suggestions
""")

# WebRTC configuration
webrtc_ctx = webrtc_streamer(
    key="grammar-check",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=8192,  # Increase the size to handle more frames
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def process_audio(audio_frames):
    """Process recorded audio frames"""
    if not audio_frames:
        st.error("No audio captured. Please try again.")
        return None
    
    temp_file_path = None
    try:
        # Create temporary file with unique name
        temp_file_path = os.path.join(tempfile.gettempdir(), f"grammar_check_{int(time.time())}.wav")
        
        # Convert and save audio
        audio_data = np.concatenate(audio_frames, axis=0).astype(np.int16)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.reshape(-1)
        sf.write(temp_file_path, audio_data, samplerate=48000)
        
        # Send to backend
        with open(temp_file_path, "rb") as f:
            response = requests.post(
                "http://localhost:5000/api/grammar-check",
                files={"audio": f},
                timeout=30
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None
    finally:
        # Ensure file is closed before deletion
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except PermissionError:
                # If still locked, schedule for deletion on program exit
                import atexit
                atexit.register(lambda: os.unlink(temp_file_path) if os.path.exists(temp_file_path) else None)

if webrtc_ctx.audio_receiver:
    st.info("🎤 Speak now (recording for 5 seconds)...")
    
    # Collect audio for 5 seconds
    audio_frames = []
    start_time = time.time()
    with st.spinner("Recording..."):
        while time.time() - start_time < 5:
            try:
                audio_frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                audio_data = audio_frame.to_ndarray()
                # Check if audio is not silent
                if np.abs(audio_data).mean() > 0.01:  # Threshold for silence detection
                    audio_frames.append(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                st.warning(f"Audio frame error: {e}")
                break
    
    if len(audio_frames) > 10:  # Require minimum 10 frames (about 0.2s of audio)
        result = process_audio(audio_frames)
        
        if result:
            st.success("Analysis complete!")
            
            # Main results
            st.header("📊 Grammar Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Text")
                st.write(result["original"])
                
                st.subheader("Corrected Text")
                st.write(result["corrected"])
                
                st.subheader("Grammar Score")
                score = result["score"]
                st.metric(label="Grammar Score", value=f"{score}/100", delta=None)
                st.progress(score / 100)
            
            with col2:
                st.subheader("Error Analysis")
                if result["errors"]:
                    st.write(f"**Number of Errors:** {len(result['errors'])}")
                    for error in result["errors"]:
                        st.markdown(f"- **Type:** {error.get('type', 'Unknown')}")
                        st.markdown(f"  **Description:** {error['description']}")
                else:
                    st.success("No grammatical errors found!")
            
            # Suggestions
            st.subheader("💡 Improvement Suggestions")
            if result["suggestions"]:
                for suggestion in result["suggestions"]:
                    st.markdown(f"- {suggestion}")
            else:
                st.info("No suggestions available.")
            
            # Detailed report
            with st.expander("📄 View Detailed Analysis Report"):
                st.markdown(result["detailed_report"])
    else:
        st.error("No speech detected or volume too low. Please speak louder.")
else:
    st.warning("Microphone is not connected. Please allow microphone access.")