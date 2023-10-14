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

## AI
aimodel = VitsModel.from_pretrained("facebook/mms-tts-eng")
aitokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

## Human
usermodel = VitsModel.from_pretrained("facebook/mms-tts-eng")
usertokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

DEBUG = False

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
parser.add_argument("-tts", "--tokenstospeak", type=int, default=3)
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

def get_user_input():
    return input("You: ")

def converse(question):
    output = llm(
        "I am %s: %s \n\nYourname is %s: %s\n\nQuestion from %s: %s\n\nAnswer:" % (
            args.username,
            args.userpersonality,
            args.ainame,
            args.aipersonality,
            args.username,
            question),
        max_tokens=0,
        temperature=0.8,
        stream=True,
        #stop=["I am %s:" % args.username, " "],
        echo=False,
    )

    def speak_line(line):
        if not line:
            return
        if DEBUG:
            print("Speaking line: %s\n" % line)

        aitext = convert_numbers_to_words(line)
        aiinputs = aitokenizer(aitext, return_tensors="pt")
        aiinputs['input_ids'] = aiinputs['input_ids'].long()

        with torch.no_grad():
            aioutput = aimodel(**aiinputs).waveform

        waveform_np = aioutput.squeeze().numpy().T
        buf = io.BytesIO()
        sf.write(buf, waveform_np, ai_sampling_rate, format='WAV')
        buf.seek(0)

        wave_obj = wave.open(buf)
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wave_obj.getsampwidth()),
                        channels=wave_obj.getnchannels(),
                        rate=wave_obj.getframerate(),
                        output=True)

        while True:
            data = wave_obj.readframes(1024)
            if not data:
                break
            stream.write(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    tokens = []

    token_count = 0
    tokens_to_speak = 0
    for item in output:
        if DEBUG:
            print("Got Item: %s\n" % json.dumps(item))
        token = item['choices'][0]['text']
        tokens.append(token)
        token_count += 1
        print("%s" % token, end='', flush=True)

        if (token_count % args.tokenstospeak == 0) and (token[len(token)-1] == ' ' or token[len(token)-1] == '\n' or token[len(token)-1] == '.'):
            line = ''.join(tokens)
            speak_line(line)
            tokens = []

    if tokens:  # if there are any remaining tokens, speak the last line
        line = ''.join(tokens)
        speak_line(line)

if __name__ == "__main__":
    initial_question = args.question
    converse(initial_question)

    while True:
        try:
            print("Press Enter to continue, or Ctrl+C to exit.")
            input()
            next_question = get_user_input()
            converse(next_question)
        except KeyboardInterrupt:
            print("\nExiting...")
            break

