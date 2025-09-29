import streamlit as st
import tempfile, os, time, logging
import numpy as np
import soundfile as sf

# External APIs
import assemblyai as aai
import google.generativeai as genai

st.set_page_config(page_title="Grammar Analysis", layout="wide")
st.title("🎤 Speech to Grammar Analysis")

st.markdown(
"""
Upload an audio file (wav/mp3/m4a) and get transcription + grammar feedback.
"""
)

# --- Load secrets (Streamlit Cloud will supply these via st.secrets) ---
ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY") or os.environ.get("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if not ASSEMBLYAI_API_KEY or not GEMINI_API_KEY:
    st.error("Missing API keys. Add ASSEMBLYAI_API_KEY and GEMINI_API_KEY in Streamlit Secrets (or set as env vars for local testing).")
    st.stop()

# Configure SDKs
aai.settings.api_key = ASSEMBLYAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    st.warning(f"Could not initialize Gemini model: {e}")
    model = None

def transcribe_with_assemblyai(audio_path):
    try:
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(speaker_labels=False)
        transcript = transcriber.transcribe(audio_path, config=config)
        return (transcript.text or "").strip()
    except Exception as e:
        logging.error(f"AssemblyAI error: {e}")
        return None

def parse_gemini_response(response, original_text):
    result = {
        "original": original_text,
        "corrected": original_text,
        "errors": [],
        "score": 0,
        "suggestions": [],
        "detailed_report": response,
    }
    try:
        if "=== CORRECTED TEXT ===" in response:
            corrected_section = response.split("=== CORRECTED TEXT ===")[1].split("===")[0].strip()
            result["corrected"] = corrected_section
        if "=== ERROR ANALYSIS ===" in response:
            error_section = response.split("=== ERROR ANALYSIS ===")[1].split("===")[0]
            error_items = [e.strip() for e in error_section.split("\n") if e.strip()]
            for item in error_items:
                result["errors"].append({"type": "Grammar Error", "description": item})
        if "Score:" in response:
            score_line = next((line for line in response.split("\n") if "Score:" in line), "Score: 0")
            digits = "".join(filter(str.isdigit, score_line))
            result["score"] = int(digits) if digits else 0
        if "=== SUGGESTIONS ===" in response:
            suggestion_section = response.split("=== SUGGESTIONS ===")[1].strip()
            result["suggestions"] = [s.strip("- ").strip() for s in suggestion_section.split("\n") if s.strip()]
    except Exception as e:
        logging.error(f"parse_gemini_response error: {e}")
    return result

def analyze_text_with_gemini(text):
    if not model:
        return {"error": "Gemini model not available"}
    prompt = f"""
You are a grammar checker. Correct the following sentence, identify grammar mistakes, and give a score from 0 to 100.

Sentence: "{text}"

Reply in this format:
Corrected: ...
=== ERROR ANALYSIS ===
- ...
=== SUGGESTIONS ===
- ...
Score: ...
"""
    try:
        response = model.generate_content(prompt).text
        return parse_gemini_response(response, text)
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return {"error": "Gemini analysis failed"}

def save_uploaded_audio(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

# ------- UI -------
uploaded = st.file_uploader("Upload a wav/mp3/m4a file", type=["wav", "mp3", "m4a", "webm"])
if uploaded:
    st.info("Saving and processing your file...")
    audio_path = save_uploaded_audio(uploaded)
    text = transcribe_with_assemblyai(audio_path)
    if not text:
        st.error("Transcription failed or no speech detected.")
    else:
        st.success("Transcribed:")
        st.write(text)
        st.info("Analyzing grammar with Gemini...")
        result = analyze_text_with_gemini(text)
        if "error" in result:
            st.error(result["error"])
        else:
            st.header("📊 Grammar Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Text")
                st.write(result["original"])
                st.subheader("Corrected Text")
                st.write(result["corrected"])
                st.subheader("Grammar Score")
                score = result["score"]
                st.metric(label="Grammar Score", value=f"{score}/100")
                st.progress(score / 100 if score else 0)
            with col2:
                st.subheader("Errors")
                if result["errors"]:
                    for e in result["errors"]:
                        st.markdown(f"- {e.get('description')}")
                else:
                    st.success("No grammar errors found")
                st.subheader("Suggestions")
                if result["suggestions"]:
                    for s in result["suggestions"]:
                        st.markdown(f"- {s}")
                else:
                    st.info("No suggestions")
            with st.expander("Detailed report"):
                st.text(result["detailed_report"])
    try:
        os.unlink(audio_path)
    except Exception:
        pass

st.caption("Local note: For local testing create .streamlit/secrets.toml with the API keys (do NOT commit that file).")
