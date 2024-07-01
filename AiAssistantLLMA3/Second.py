import requests
import json
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import pygame
import time
import sys
import contextlib
import threading

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}

istoric_conversatie = []
stop_event = threading.Event()
tts_lock = threading.Lock()

def genereaza_raspuns(prompt):
    istoric_conversatie.append(prompt)

    prompt_complet = "\n".join(istoric_conversatie)

    date = {
        "model": "llama3",
        "stream": False,
        "prompt": prompt_complet,
    }

    raspuns = requests.post(url, headers=headers, data=json.dumps(date))

    if raspuns.status_code == 200:
        text_raspuns = raspuns.text
        date = json.loads(text_raspuns)
        raspuns_actual = date["response"]
        istoric_conversatie.append(raspuns_actual)
        return raspuns_actual
    else:
        print("Eroare:", raspuns.status_code, raspuns.text)
        return None

def vorbeste_text(text, lang='ro'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        cale_temp = fp.name
        tts.save(cale_temp)
        print(f"Salvat TTS la {cale_temp}")

    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            pygame.mixer.init()
            pygame.mixer.music.load(cale_temp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.1)
            pygame.mixer.music.unload()
    os.remove(cale_temp)
    print(f"Șters fișierul TTS {cale_temp}")

def obtine_intrare_audio(lang='ro-RO'):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Pregătește-te să vorbești...")
        time.sleep(2)
        print("Ascult...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5)

    try:
        print("Recunoaștere...")
        text = recognizer.recognize_google(audio, language=lang)
        print(f"Ai spus: {text}")
        return text
    except sr.UnknownValueError:
        print("Îmi pare rău, nu am înțeles asta.")
        return None
    except sr.RequestError as e:
        print(f"Nu am putut solicita rezultatele; {e}")
        return None

def conversation_thread():
    while not stop_event.is_set():
        intrare_utilizator = obtine_intrare_audio()
        
        if intrare_utilizator and intrare_utilizator.lower() in ['exit', 'quit', 'ieșire', 'părăsiți', 'închide']:
            print("Închid conversația.")
            with tts_lock:
                vorbeste_text("Închid conversația.", lang='ro')
            stop_event.set()
            break

        if intrare_utilizator:
            raspuns = genereaza_raspuns(intrare_utilizator)
            if raspuns:
                print("Răspuns AI:", raspuns)
                with tts_lock:
                    vorbeste_text(raspuns, lang='ro')

if __name__ == "__main__":
    conversation_thread = threading.Thread(target=conversation_thread)
    conversation_thread.start()

    try:
        while conversation_thread.is_alive():
            conversation_thread.join(1)
    except KeyboardInterrupt:
        stop_event.set()
        conversation_thread.join()
        print("Conversația a fost oprită.")