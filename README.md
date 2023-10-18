```
Chatbot that speaks

Multispeaker LLM locally with TTS all free if run locally

Chris Kennedy (C) The Groovy Organization Apache License
```
# Conscious Chat: Multispeaker LLM with TTS

This chatbot, developed by Chris Kennedy of The Groovy Organization, creates an engaging conversation environment through the use of multispeaker Language Models (LLM) and Text-to-Speech (TTS) technology. The goal of the project is to foster multi-user interactions among multiple AI entities, creating a Game of Life-like phenomenon for human entertainment and question answering.

This project is licensed under the Apache License.

## Features

- **Multispeaker LLM**: Engage in conversation with AI entities possessing unique personalities.
- **Text-to-Speech (TTS)**: Hear responses audibly, adding a layer of realism to the interaction.
- **Customizable Personalities**: Tailor the personality of AI and user entities to your liking.
- **Local Execution**: All functionalities are available for free when run locally.
- **Auto Generate Responses**: The chatbot can automatically generate responses without requiring manual input.
- **Stop on New Line**: The conversation can be configured to pause when a new line is detected.
- **Stochastic Token Generation**: The chatbot can generate a variable number of tokens per response, making the conversation feel more dynamic and natural.
- **Stream Speak**: Streams the conversation per `tokenstospeak` value instead of speaking at new lines.
- **Episode Mode**: Enables the output to follow a TV episode script format.
- **Multi-Lingual**: Ability to speak in most languages on output text and voice (if uroman.pl is installed).
- **Webpage Retrieval**: Ability to look up all URLs in prompt and load up the LLM context with the summary + store them locally for future fast retrieval.

## TODO

- DONE! Thread the text to speech generation and speaking to allow LLM text generation simultaneously for more efficient usage and speed asyncronously.
- Prompt template usage for cleaning up the prompt and making it standard.
- Document retrieval from PDFs, Json, Text files for Context injection.
- Female TTS model voices.
- Better TTS quality? Facebook MMS is okay, nice natural sound, yet sort of rough sounding and 16k sample rate. Can we upscale it???
- Document retrieval tuning of settings, speed up if possible.
- Stable Diffusion image generation for background with LLM text on top of it.
- Background Music generation.
- Twitch Chat integration with users able to configure their own chatbot, control current ones, speak to them.
- History per Twitch user and persistent history in general per chatbot.
- Multiple models running simultaneously each as a personality chatting with other running models and sharing context and history.
- News context injection for news reporting and seeding story generation without human interaction.
- Storage of sessions in a DB (long-term, not a priority).
- Generate a video in HLS or MpegTS format to stream out and push up to YouTube.
- 3D Model control for characters vtuber type Blender models controlled by text generation and image generation.
- Timing metrics
- Many Many odds and ends, fixups and optimizations.
- Have ability to run and stream on Twitch with users in chat interacting with the chat and multi-modal mpegts stream generation (can we rtmp up directly?)

## Recommended Models

- TTS: [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)
    - https://huggingface.co/docs/transformers/main/en/model_doc/vits More details on config options
    - More information: [Meta AI Launches Massively Multilingual Speech MMS Project](https://www.marktechpost.com/2023/05/30/meta-ai-launches-massively-multilingual-speech-mms-project-introducing-speech-to-text-text-to-speech-and-more-for-1000-languages/)
    - Research paper: https://arxiv.org/abs/2106.06103
- LLM: [TheBloke/zephyr-7B-alpha-GGUF](https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF)
    - More information: https://news.ycombinator.com/item?id=37842618
- Summarization:
    - https://huggingface.co/kunaltilaganji/Abstractive-Summarization-with-Transformers
- Embeddings: LLama.cpp is slow
    - https://huggingface.co/Aryanne/OpenLlama-Platypus-3B-gguf
    - https://platypus-llm.github.io/

## Setup and Usage

### Prerequisites

- Python 3.x
- Python framework for llama.cpp from https://github.com/abetlen/llama-cpp-python
- Install the required packages from `requirements.txt` with pip:
- Uroman Perl program https://github.com/isi-nlp/uroman.git (if wanting multi-lingual speaking in more than English)
    - Set the path to uroman.pl, where you cloned it to: "export UROMAN=`pwd`/uroman/bin".
    - Use the "-ro True" option to romanize the TTS speaking text.

```bash
# Development SDK on Mac OS X

# Python3 path set on Mac OS X to SDK tools

# Install OpenSSL on Mac OS X so Python doesn't complain
brew install openssl

# Portaudio on Mac OS X
brew install portaudio

# setup Python virtual environment
pip install virtualenv
python -m venv consciousChat
source consciousChat/bin/activate

# Install required dependency python3 packages
pip install -r requirements.txt

# Run program
./characterChat.py

# Stop env and clean it up
deactivate
```

### Running the Chatbot

You can initiate a chat session by running the `characterChat.py` script from your terminal:

```bash
./characterChat.py [options]
```

#### Options

```plaintext
  -h, --help                        Show this help message and exit
  -m, --model MODEL                 Path to the model
  -ag, --autogenerate               Enables automatic response generation
  -sonl, --stoponnewline            Stops the conversation on a new line
  -q, --question QUESTION           Question to ask the AI
  -un, --username USERNAME          Username
  -up, --userpersonality USERPERSONALITY     User Personality description
  -ap, --aipersonality AIPERSONALITY         AI Personality description
  -an, --ainame AINAME              AI Name
  -asr, --aispeakingrate AISPEAKINGRATE      AI Speaking rate
  -ans, --ainoisescale AINOISESCALE         AI Noise scale
  -apr, --aisamplingrate AISAMPLINGRATE      AI Sampling rate
  -usr, --userspeakingrate USERSPEAKINGRATE  User Speaking rate
  -uns, --usernoisescale USERNOISESCALE      User Noise scale
  -upr, --usersamplingrate USERSAMPLINGRATE  User Sampling rate
  -tts, --tokenstospeak TOKENSTOSPEAK       Minimum number of tokens to speak
  -sts, --stoptokens STOPTOKENS     Specific tokens at which to stop speaking
  -ctx, --context CONTEXT           Context window size for the LLM
  -mt, --maxtokens MAXTOKENS        Maximum tokens for response
  -d, --debug                       Enables debug print statements
  -s, --silent                      Disables speaking the AI's responses
  -e, --episode                     Enables Episode mode
  -pc, --promptcompletion           Customizable prompt completion
  -re, --roleenforcer               Customizable role enforcer statement
  -l, --language                    Output Text and Speech in another language
```

## Contributing

Feel free to fork the project, open a PR, or submit issues with suggestions, corrections, or improvements.

For more information, please contact Chris Kennedy at The Groovy Organization.

## License

This project is under the Apache License and is maintained by Chris Kennedy of The Groovy Organization.

## Try it out!

Embark on a journey of interactive conversations, rich auditory experiences, and explore the realms of AI-human interactions with Conscious Chat.
