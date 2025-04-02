import streamlit as st
import speech_recognition as sr
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# NVIDIA Nemotron-70B API setup
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=st.secrets.KEYS.NVIDIA_API_KEY  # Replace with your actual API key
)

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(temp_audio.name, "wb") as f:
        f.write(audio.get_wav_data())
    
    return temp_audio.name

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)  # Using Google STT for now
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Error with the speech recognition service"

def generate_response(user_input):
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[{"role": "system", "content": "Provide short and precise answers. Avoid unnecessary details."},
            {"role": "user", "content": user_input}],
        temperature=0.3,
        top_p=0.7,
        max_tokens=100,
        stream=True
    )
    
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
    
    return response_text

def main():
    st.title("Real-Time Voice Assistant")
    if st.button("Start Listening"):
        audio_file = record_audio()
        text = speech_to_text(audio_file)
        st.write(f"You said: {text}")
        os.remove(audio_file)  # Clean up temp file
        
        if text:
            response = generate_response(text)
            st.write(f"Assistant: {response}")

if __name__ == "__main__":
    main()
