import time  # Import the time module to measure time
import threading
from nicegui import ui, events

start_time = time.time()  # Record the start time

from groq import Groq
from PIL import ImageGrab, Image
from openai import OpenAI
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import torch
import pyperclip
import cv2
import pyaudio
import os
import re
import noisereduce as nr
import numpy as np
from scipy.io import wavfile
from PIL import Image

wake_word = 'Bob'
groq_client = Groq(api_key="")
genai.configure(api_key='')
openai_client = OpenAI(api_key='')
web_cam = cv2.VideoCapture(0)

sys_msg = (
    'You are a multi-modal AI voice Assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity. You will also only make responses in Romanian'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

num_cores = os.cpu_count()
whisper_size = 'small'  # Use the large-v2 model for highest accuracy
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',        # Utilizare 'cpu' deoarece GPU nu este suportat
    cpu_threads=num_cores, # Utilizează toate firele CPU disponibile
    num_workers=num_cores  # Utilizează toți lucrătorii disponibili
)

r = sr.Recognizer()
source = sr.Microphone()

# Initialize NiceGUI elements
response_output = None
image_output = None
console_output = None

# Function to update the response label
def update_response(response):
    response_output.set_text(response)
    console_output.set_text(console_output.text + f'\nASSISTANT: {response}')

# Function to update the captured image
def update_image(image_path):
    image_output.set_source(image_path)

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    start_time = time.time()
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    end_time = time.time()
    response = chat_completion.choices[0].message
    convo.append(response)
    print(f"Groq prompt execution time: {end_time - start_time:.2f} seconds")
    console_output.set_text(console_output.text + f'\nGroq prompt execution time: {end_time - start_time:.2f} seconds')
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed. \n'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    
    start_time = time.time()
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    end_time = time.time()
    response = chat_completion.choices[0].message
    print(f"Function call execution time: {end_time - start_time:.2f} seconds")
    console_output.set_text(console_output.text + f'\nFunction call execution time: {end_time - start_time:.2f} seconds')
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    update_image(path)  # Update the image in the UI

def web_cam_capture():
    web_cam = cv2.VideoCapture(0)
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        return
    ret, frame = web_cam.read()
    path = 'webcam.jpg'
    if ret:
        cv2.imwrite(path, frame)
        update_image(path)  # Update the image in the UI
    web_cam.release()

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No clipboard text to copy")
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semtantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    start_time = time.time()
    response = model.generate_content([prompt, img])
    end_time = time.time()
    print(f"Vision prompt execution time: {end_time - start_time:.2f} seconds")
    console_output.set_text(console_output.text + f'\nVision prompt execution time: {end_time - start_time:.2f} seconds')
    return response.text

def speak(text):
    player = pyaudio.PyAudio()
    stream = player.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='onyx',
        response_format='pcm',
        input=text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    stream.write(chunk)
                    stream_start = True
    stream.stop_stream()
    stream.close()
    player.terminate()

def wav_to_text(audio_path):
    start_time = time.time()
    segments, _ = whisper_model.transcribe(audio_path, language='ro')  # Specifică limba română
    end_time = time.time()
    text = ''.join(segment.text for segment in segments)
    print(f"Audio transcription time: {end_time - start_time:.2f} seconds")
    console_output.set_text(console_output.text + f'\nAudio transcription time: {end_time - start_time:.2f} seconds')
    return text

def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    # Load the audio file
    rate, data = wavfile.read(prompt_audio_path)

    # Save the original audio to file (no noise reduction)
    wavfile.write(prompt_audio_path, rate, data.astype(np.int16))
    
    # Transcribe audio to text
    start_time = time.time()
    prompt_text = wav_to_text(prompt_audio_path)
    end_time = time.time()
    console_output.set_text(console_output.text + f'\nAudio transcription time: {end_time - start_time:.2f} seconds')

    # Extract the clean prompt using the wake word
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt: 
        console_output.set_text(console_output.text + f'\nUSER: {clean_prompt}')
        start_time = time.time()
        call = function_call(clean_prompt)
        end_time = time.time()
        console_output.set_text(console_output.text + f'\nFunction call decision time: {end_time - start_time:.2f} seconds')
        
        if 'take screenshot' in call:
            take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            web_cam_capture()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam.jpg')
        elif 'extract clipboard' in call:
            clipboard_content = get_clipboard_text()
            clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {clipboard_content}'
            visual_context = None
        else:
            visual_context = None

        # Generate response from the AI model
        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        update_response(response)
        
        # Speak out the response
        speak(response)
        end_time = time.time()
        print(f"Text-to-speech processing time: {end_time - start_time:.2f} seconds")
        console_output.set_text(console_output.text + f'\nText-to-speech processing time: {end_time - start_time:.2f} seconds')

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('Ziceti ', wake_word, 'urmat de promptul dvs. \n')
    console_output.set_text(console_output.text + f'\nZiceti {wake_word} urmat de promptul dvs. \n')
    r.listen_in_background(source, callback)
    
    while True:
        time.sleep(.5)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

def text_input_callback(event):
    input_text = event.value
    if input_text:
        print(f'USER (typed): {input_text}')
        console_output.set_text(console_output.text + f'\nUSER (typed): {input_text}')
        start_time = time.time()
        call = function_call(input_text)
        end_time = time.time()
        print(f"Function call decision time: {end_time - start_time:.2f} seconds")
        console_output.set_text(console_output.text + f'\nFunction call decision time: {end_time - start_time:.2f} seconds')
        
        visual_context = None

        start_time = time.time()
        response = groq_prompt(prompt=input_text, img_context=visual_context)
        end_time = time.time()
        print(f"Total Groq prompt processing time: {end_time - start_time:.2f} seconds")
        console_output.set_text(console_output.text + f'\nTotal Groq prompt processing time: {end_time - start_time:.2f} seconds')
        
        print(f'ASSISTANT: {response}')
        update_response(response)
        start_time = time.time()
        speak(response)
        end_time = time.time()
        print(f"Text-to-speech processing time: {end_time - start_time:.2f} seconds")
        console_output.set_text(console_output.text + f'\nText-to-speech processing time: {end_time - start_time:.2f} seconds')

        # Clear the input field
        event.sender.value = ''

# Start the voice assistant in a separate thread
def start_voice_assistant():
    start_listening()


# Create a NiceGUI interface
def create_interface():
    global response_output, image_output, console_output

    with ui.column():
        with ui.row().style('justify-content: space-between; align-items: center;'):
            ui.label('AI Voice Assistant').classes('text-2xl font-bold')

        with ui.column():
            ui.label('Console Output:')
            console_output = ui.label('').classes('text-mono bg-gray-100 p-2')

        ui.label('Type your input below:')
        input_field = ui.input().classes('w-full p-2')

        # Adjusting the row to use flexbox for better alignment
        with ui.row().style('display: flex; align-items: center; justify-content: space-around;'):
            ui.button('Submit', on_click=lambda: text_input_callback(input_field.value)).classes('button-common')
            ui.button('Start Listening', on_click=lambda: threading.Thread(target=start_voice_assistant).start()).classes('button-common')
            ui.button('Take Screenshot', on_click=take_screenshot).classes('button-common')
            ui.button('Capture Webcam', on_click=web_cam_capture).classes('button-common')

        with ui.row():
            ui.label('Assistant Response:')
            response_output = ui.label('Waiting for response...').classes('text-lg')

        ui.label('Captured Image:')
        image_output = ui.image().style('max-width: 400px; max-height: 300px;')

        ui.image(r'C:\Users\luisg\OneDrive\Desktop\Assistent\Sigla8cp.png').style('width: 300px; height: auto;')
        ui.label('Proiect realizat de:')
        ui.label('Ghitu Luis-Federico')
        ui.label('Georgescu Marius-Mihai')
        ui.label('Mihai Marian-Eduard-Virgil')

        start_time = time.time()
        end_time = time.time()
        console_output.set_text(f'Application load time: {end_time - start_time:.2f} seconds')
        


def update_image(path):
    if image_output:
        image_output.set_source(path)  # Assuming 'set_src' is the correct method to update the image source
        print(f"Image source updated to {path}")  # Debug statement to confirm the path update

def text_input_callback(input_text):
    if input_text:
        console_output.set_text(console_output.text + f'\nUSER (typed): {input_text}')
        start_time = time.time()
        call = function_call(input_text)
        end_time = time.time()
        console_output.set_text(console_output.text + f'\nFunction call decision time: {end_time - start_time:.2f} seconds')
        
        visual_context = None

        if 'take screenshot' in call:
            print('Taking screenshot')
            take_screenshot()
            visual_context = vision_prompt(prompt=input_text, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print('Capturing webcam')
            web_cam_capture()
            visual_context = vision_prompt(prompt=input_text, photo_path='webcam.jpg')
        elif 'extract clipboard' in call:
            print('Copying clipboard text')
            paste = get_clipboard_text()
            input_text = f'{input_text}\n\n CLIPBOARD CONTENT: {paste}'
        
        response = groq_prompt(prompt=input_text, img_context=visual_context)
        console_output.set_text(console_output.text + f'\nTotal Groq prompt processing time: {end_time - start_time:.2f} seconds')
        
        print(f'ASSISTANT: {response}')
        response_output.set_text(response)
        speak(response)

# Run NiceGUI
create_interface()
ui.run()