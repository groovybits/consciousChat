#!/usr/local/bin/python3


"""
Chris Kennedy (C) 2023 The Groovy Organization
Apache license

Chatbot that speaks
"""

import json
import argparse
from transformers import VitsModel, AutoTokenizer, set_seed
import torch
import soundfile as sf
import io
import inflect
import re
from llama_cpp import Llama
import sounddevice as sd
import pyaudio
import wave
import os


aimodel = VitsModel.from_pretrained("facebook/mms-tts-eng")
aitokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
usermodel = VitsModel.from_pretrained("facebook/mms-tts-eng")
usertokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

def convert_numbers_to_words(text):
    p = inflect.engine()

    def num_to_words(match):
        number = match.group()
        if '.' in number:
            parts = number.split('.')
            words = f"{p.number_to_words(parts[0])} point {p.number_to_words(parts[1])}"
        else:
            words = p.number_to_words(number)
        return words

    text_with_words = re.sub(r'\b\d+(\.\d+)?\b', num_to_words, text)
    return text_with_words

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-alpha.Q8_0.gguf")
parser.add_argument("-q", "--question", type=str, default="How has your day been")
parser.add_argument("-un", "--username", type=str, default="Human")
parser.add_argument("-up", "--userpersonality", type=str, default="I am a magical girl from an anime that is here to talk to other magical girls")
parser.add_argument("-ap", "--aipersonality", type=str, default="You are a magical girl from an anime that is here to help however needed.")
parser.add_argument("-an", "--ainame", type=str, default="Usagi")
parser.add_argument("-mt", "--maxtokens", type=int, default=0)
parser.add_argument("-asr", "--aispeakingrate", type=float, default=1.2)
parser.add_argument("-ans", "--ainoisescale", type=float, default=1.0)
parser.add_argument("-apr", "--aisamplingrate", type=int, default=aimodel.config.sampling_rate)
parser.add_argument("-usr", "--userspeakingrate", type=float, default=0.8)
parser.add_argument("-uns", "--usernoisescale", type=float, default=1.0)
parser.add_argument("-upr", "--usersamplingrate", type=int, default=usermodel.config.sampling_rate)
args = parser.parse_args()

ai_speaking_rate = args.aispeakingrate
ai_noise_scale = args.ainoisescale
ai_sampling_rate = args.aisamplingrate
aimodel.speaking_rate = ai_speaking_rate
aimodel.noise_scale = ai_noise_scale

user_speaking_rate = args.userspeakingrate
user_noise_scale = args.usernoisescale
user_sampling_rate = args.usersamplingrate
usermodel.speaking_rate = user_speaking_rate
usermodel.noise_scale = user_noise_scale

llm = Llama(model_path=args.model, n_ctx=32768)

output = llm(
        "I am %s: %s \n\nYourname is %s: %s\n\nQuestion from %s: %s\n\nAnswer:" % (
            args.username,
            args.userpersonality,
            args.ainame,
            args.aipersonality,
            args.username,
            args.question),
    max_tokens=0,
    temperature=0.8,
    stream=True,
    #stop=["I am %s:" % args.username, " "],
    echo=False,
)

def speak_line(line):
    if not line:
        # If line is empty, do nothing and return early
        return
    print("Speaking line: %s\n" % line)

    # Convert numbers in the line to words
    aitext = convert_numbers_to_words(line)

    # Tokenize the text for the model
    aiinputs = aitokenizer(aitext, return_tensors="pt")

    # Ensure input_ids is of data type Long
    aiinputs['input_ids'] = aiinputs['input_ids'].long()

    # Generate the audio waveform
    with torch.no_grad():
        aioutput = aimodel(**aiinputs).waveform

    # Convert the waveform to a NumPy array and transpose it
    waveform_np = aioutput.squeeze().numpy().T

    # Write the waveform to a BytesIO buffer
    buf = io.BytesIO()
    sf.write(buf, waveform_np, ai_sampling_rate, format='WAV')
    buf.seek(0)

    # Open the waveform with the wave module
    wave_obj = wave.open(buf)

    # Open a PyAudio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wave_obj.getsampwidth()),
                    channels=wave_obj.getnchannels(),
                    rate=wave_obj.getframerate(),
                    output=True)

    # Play the audio stream
    while True:
        data = wave_obj.readframes(1024)
        if not data:
            break
        stream.write(data)

    # Stop the stream, close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

# Initialize an empty list to accumulate tokens
tokens = []

# Iterate through the output generator
for item in output:
    print("Got Item: %s\n" % json.dumps(item))
    # Extract the token text from the item
    token = item['choices'][0]['text']

    # Assume newline token is represented as '\n'
    if token == '\n':
        # Join the accumulated tokens to form a line
        line = ''.join(tokens)
        # Speak the line
        speak_line(line)
        # Reset the tokens list for the next line
        tokens = []
    else:
        # Accumulate the tokens
        tokens.append(token)

# Handle any remaining tokens (if the text doesn't end with a newline)
if tokens:
    line = ''.join(tokens)
    speak_line(line)
