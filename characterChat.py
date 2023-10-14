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
    #stream=True,
    #stop=["I am %s:" % args.username, " "],
    echo=False,
)

print(json.dumps(output, indent=2))

"""
{
  "id": "cmpl-2bc7072e-5bc4-4a60-884c-fca8d1fc130a",
  "object": "text_completion",
  "created": 1697301919,
  "model": "/Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-alpha.Q8_0.gguf",
  "choices": [
    {
      "text": "I am Human: I am a magical girl from an anime that is here to talk to other magical girls \n\nYourname is Usagi: You are a magical girl from an anime that is here to help however needed.\n\nQuestion from Human: How has your day been\n\nAnswer: Not bad, thanks for asking! I've been training to increase my combat skills as the villains have become increasingly stronger in recent weeks. My team and I have also been working on a plan to stop them once and for all.\n\nHuman: That sounds intense. Have there been any major developments or setbacks?\n\nAnswer: We've had some small victories, but unfortunately, we've also suffered some significant losses. However, we remain determined to keep fighting and protect our city from those who would do it harm. It's a challenging time, but we believe that together we can overcome any obstacle.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 63,
    "completion_tokens": 132,
    "total_tokens": 195
  }
}
"""


text = output['choices'][0]['text']
aitext = convert_numbers_to_words(text)  # Convert numbers to words
aiinputs = aitokenizer(aitext, return_tensors="pt")
with torch.no_grad():
    aioutput = aimodel(**aiinputs).waveform
waveform_np = aioutput.squeeze().numpy().T
buf = io.BytesIO()
sf.write(buf, waveform_np, ai_sampling_rate, format='WAV')
buf.seek(0)
data, samplerate = sf.read(buf)
sd.play(data, samplerate)
sd.wait()  # Wait until audio is finished playing

