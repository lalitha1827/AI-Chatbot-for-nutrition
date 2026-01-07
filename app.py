'''
+-------------------+        +-----------------------+        +------------------+        +------------------------+
|   Step 1: Install |        |  Step 2: Real-Time    |        |  Step 3: Pass    |        |  Step 4: Live Audio    |
|   Python Libraries|        |  Transcription with   |        |  Real-Time       |        |  Stream from ElevenLabs|
+-------------------+        |       Google STT      |        +------------------+        +------------------------+
|                   |        +-----------------------+        |      OpenAI      |        +------------------------+
| - assemblyai      |                    |                    +------------------+                    |
| - openai          |                    |                             |                              |
| - elevenlabs      |                    v                             v                              v
| - mpv             |        +-----------------------+        +------------------+        +------------------------+
| - portaudio       |        |                       |        |                  |        |                        |
+-------------------+        |  Google STT performs  |-------->  OpenAI generates|-------->  ElevenLabs streams   |
                             |  real-time speech-to- |        |  response based  |        |  response as live      |
                             |  text transcription   |        |  on transcription|        |  audio to the user     |
                             |                       |        |                  |        |                        |
                             +-----------------------+        +------------------+        +------------------------+

###### Step 1: Install Python libraries ######


# Install instructions (run these in your terminal, not in the script):
# pip install pipwin
# pipwin install pyaudio
# pip install speechrecognition
# pip install elevenlabs==0.3.0b0
# pip install --upgrade openai
# For mpv on Windows, download from https://mpv.io/installation/ and add to PATH if needed
'''
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
import requests
import re

class AI_Assistant:
    def __init__(self):
        self.gemini_api_key = "google_api_key"  # <-- Replace with your Gemini API key
        self.elevenlabs_api_key = "elevenlabs_api_key"
        self.full_transcript = [
            {"role":"system", "content":"You are a voice-interactive personalized diet planner. Your role is to ask users a series of health and dietary questions, understand their responses, and generate a custom diet plan tailored to their needs. The plan should include breakfast, lunch, and dinner suggestions, each with 2 and 3 items and approximate calories per item. Be clear, concise, and ensure the recommendations match the user's health history, goals, and food preferences."},
        ]
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = False

    def start_transcription(self):
        print("\nReal-time transcription started. Speak into your microphone.")
        self.listening = True
        while self.listening:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source)
            try:
                transcript = self.recognizer.recognize_google(audio)
                print(f"Patient: {transcript}")
                self.generate_ai_response(transcript)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

    def stop_transcription(self):
        self.listening = False

    def get_first_gemini_model(self):
        url = "https://generativelanguage.googleapis.com/v1/models"
        headers = {"x-goog-api-key": self.gemini_api_key}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()
            print("[Gemini API available models]", result)
            # Only return models that support 'generateContent'
            for model in result.get("models", []):
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    return model["name"]
        except Exception as e:
            print(f"[Gemini API list models error]: {e}")
        return None

    def generate_ai_response(self, transcript):
        self.full_transcript.append({"role":"user", "content": transcript})
        # Prepare prompt for Gemini
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.full_transcript])
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        # Use Gemini 1.5 Flash model explicitly
        model_name = "gemini-1.5-flash"
        endpoint = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            print("[Gemini API raw response]", result)  # Debug print
            ai_response = result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            ai_response = f"[Gemini API error]: {e}"
        self.generate_audio(ai_response)
        self.full_transcript.append({"role":"assistant", "content": ai_response})

    def generate_audio(self, text):
        print(f"\nAI Receptionist: {text}")
        try:
            client = ElevenLabs(api_key=self.elevenlabs_api_key)
            voices_response = client.voices.get_all()
            voices = voices_response.voices if hasattr(voices_response, 'voices') else []
            voice_id = voices[0].voice_id if voices else None
            audio_gen = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )
            audio = b''.join(audio_gen)
            # Save and play audio for local use
            with open("output.mp3", "wb") as f:
                f.write(audio)
            try:
                import subprocess
                subprocess.run(["mpv", "output.mp3"])
            except Exception as e:
                print(f"[Error playing audio locally]: {e}")
            return audio
        except Exception as e:
            print(f"[Error in generate_audio]: {e}")
            return b""

    def clean_markdown(self, text: str) -> str:
        # Remove bold, italics, inline code, and links
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)        # italics
        text = re.sub(r'`([^`]*)`', r'\1', text)          # inline code
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)  # links
        return text

    def get_response_and_audio(self, user_text: str) -> tuple[str, bytes]:
        # Add user message to transcript
        self.full_transcript.append({"role": "user", "content": user_text})
        # Prepare prompt for Gemini
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.full_transcript])
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        model_name = "gemini-1.5-flash"
        endpoint = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent"
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            ai_response = result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            ai_response = f"[Gemini API error]: {e}"
        self.full_transcript.append({"role": "assistant", "content": ai_response})
        # Clean markdown from AI response before TTS
        clean_response = self.clean_markdown(ai_response)
        # Use ElevenLabs for TTS and always return audio bytes
        try:
            audio_bytes = self.generate_audio(clean_response)
        except Exception as e:
            print(f"[Error in ElevenLabs TTS]: {e}")
            audio_bytes = b""
        print(f"[DEBUG] ElevenLabs audio_bytes length: {len(audio_bytes)}")
        return ai_response, audio_bytes

















