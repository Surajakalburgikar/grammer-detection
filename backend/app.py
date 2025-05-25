import os
import logging
import tempfile
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import assemblyai as aai
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Set up APIs
aai.settings.api_key = ASSEMBLYAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
try:
    # Explicitly load the gemini-2.0-flash model
    model = genai.GenerativeModel("gemini-2.0-flash")
    logging.info("Gemini model 'gemini-2.0-flash' initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Gemini model 'gemini-2.0-flash': {e}")
    model = None

# Flask setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)


@app.route("/api/grammar-check", methods=["POST"])
def grammar_check():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Create unique temp file path
    temp_file_path = os.path.join(tempfile.gettempdir(), f"grammar_audio_{int(time.time())}.wav")

    try:
        # Save audio file
        audio_file = request.files["audio"]
        audio_file.save(temp_file_path)

        # Verify file size is reasonable (not empty)
        if os.path.getsize(temp_file_path) < 1024:  # Less than 1KB
            return jsonify({"error": "Audio file too small - no speech detected"}), 400

        # Transcribe
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(speaker_labels=False)
        transcript = transcriber.transcribe(temp_file_path, config=config)

        if not transcript.text or len(transcript.text.strip()) < 3:
            return jsonify({"error": "No speech detected or speech too short"}), 400

        original_text = transcript.text.strip()

        # Grammar analysis
        if not model:
            return jsonify({"error": "Gemini model not initialized"}), 500

        prompt = f"""
        You are a grammar checker. Correct the following sentence, identify grammar mistakes, and give a score from 0 to 100.

        Sentence: "{original_text}"

        Reply in this format:
        Corrected: ...
        Errors:
        - error 1
        - error 2
        Score: ...
        """

        response = model.generate_content(prompt).text
        result = parse_gemini_response(response, original_text)

        return jsonify(result)

    except aai.error.InvalidParameterError:
        return jsonify({"error": "Invalid audio file format"}), 400
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": "Processing error"}), 500
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logging.warning(f"Failed to delete temp file: {e}")


def parse_gemini_response(response, original_text):
    """Parse Gemini's response into structured data"""
    result = {
        "original": original_text,
        "corrected": original_text,
        "errors": [],
        "score": 0,
        "suggestions": [],
        "detailed_report": response,
    }

    try:
        # Extract corrected text
        if "=== CORRECTED TEXT ===" in response:
            corrected_section = response.split("=== CORRECTED TEXT ===")[1].split("===")[0].strip()
            result["corrected"] = corrected_section

        # Extract errors
        if "=== ERROR ANALYSIS ===" in response:
            error_section = response.split("=== ERROR ANALYSIS ===")[1].split("===")[0]
            error_items = [e.strip() for e in error_section.split("\n") if e.strip()]
            for item in error_items:
                if item.startswith("Incomplete sentence") or item.startswith("-"):
                    error = {
                        "type": "Grammar Error",
                        "description": item,
                    }
                    result["errors"].append(error)

        # Extract score
        if "Score:" in response:
            score_line = next((line for line in response.split("\n") if "Score:" in line), "Score: 0")
            result["score"] = int("".join(filter(str.isdigit, score_line)))

        # Extract suggestions
        if "=== SUGGESTIONS ===" in response:
            suggestion_section = response.split("=== SUGGESTIONS ===")[1].strip()
            result["suggestions"] = [s.strip("- ").strip() for s in suggestion_section.split("\n") if s.strip()]

    except Exception as e:
        logging.error(f"Error parsing response: {e}")

    return result


if __name__ == "__main__":
    app.run(debug=True)