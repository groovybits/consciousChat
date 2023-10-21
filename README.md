```
Completely local LLM TTS Stable Diffusion Chatbot that speaks/writes any language with images and subtitles.

Multispeaker LLM locally with TTS all free if run locally. GPT-3.5 LLM Twitch bot streamer.

Chris Kennedy (C) The Groovy Organization Apache License

```
# Conscious Chat: Local Free to run 24/7 Multispeaker/Multi-lingual LLM with TTS Twitch Bot!

This chatbot, developed by Chris Kennedy of The Groovy Organization, creates an engaging conversation environment through the use of multispeaker Language Models (LLM) and Text-to-Speech (TTS) technology. The goal of the project is to foster multi-user interactions among multiple AI entities, creating a Game of Life-like phenomenon for human entertainment and question answering.

Try it at [Groovy AI Bot on Twitch](https://twitch.com/groovyaibot) to see it in action!

This project is licensed under the Apache License. You can use it to build your Twitch AI chatbot empire.

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
- **Twitch Bot**: Ability to have twitch users interact with the conversation via Twitch chatrooms and an OBS. Future output will be direct to twitch as video.
- **User History**: Bot has knowlege of users past questions to build up a context and impression of them.
- **Stable Diffusion Images**: Images from output speaker lines, 6-10 a minute + capable with tokens driving generation by sentence chunking.
- **Hard Subtitle**: Subtitles synced with audio speaking voice and image generation.

## TODO

- Prompt template usage for cleaning up the prompt and making it standard.
- Document retrieval from PDFs, Json, Text files for Context injection.
- Female TTS model voices.
- Document retrieval tuning of settings, speed up if possible.
- Background Music generation.
- Persistent chat history between sessions.
- Multiple models running simultaneously each as a personality chatting with other running models and sharing context and history.
- News context injection for news reporting and seeding story generation without human interaction.
- Storage of sessions in a DB (long-term, not a priority).
- Generate a video in HLS or MpegTS format to stream out and push up to YouTube.
- 3D Model control for characters vtuber type Blender models controlled by text generation and image generation.
- Timing metrics
- Many Many odds and ends, fixups and optimizations.

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
usage: characterChat.py [-h] [-l LANGUAGE] [-pd PERSISTDIRECTORY] [-m MODEL] [-em EMBEDDINGMODEL] [-ag] [-ss] [-tts TOKENSTOSPEAK]
                        [-aittss AITTSSEED] [-usttss USTTSSEED] [-mtts MINTOKENSTOSPEAK] [-q QUESTION] [-un USERNAME]
                        [-up USERPERSONALITY] [-ap AIPERSONALITY] [-an AINAME] [-asr AISPEAKINGRATE] [-ans AINOISESCALE]
                        [-apr AISAMPLINGRATE] [-usr USERSPEAKINGRATE] [-uns USERNOISESCALE] [-upr USERSAMPLINGRATE] [-sts STOPTOKENS]
                        [-ctx CONTEXT] [-mt MAXTOKENS] [-gl GPULAYERS] [-t TEMPERATURE] [-d] [-dd] [-s] [-ro] [-e]
                        [-pc PROMPTCOMPLETION] [-re ROLEENFORCER] [-sd] [-udb URLSDB] [-cdb CHATDB] [-ectx EMBEDDINGSCONTEXT]
                        [-ews EMBEDDINGWINDOWSIZE] [-ewo EMBEDDINGWINDOWOVERLAP] [-eds EMBEDDINGDOCSIZE] [-hctx HISTORYCONTEXT]
                        [-im IMAGEMODEL] [-ns] [-tw] [-gu] [-si] [-ll LOGLEVEL] [-ars AUDIOPACKETREADSIZE] [-ren] [-wi WIDTH]
                        [-he HEIGHT] [-as]

optional arguments:
  -h, --help            show this help message and exit
  -l LANGUAGE, --language LANGUAGE
                        Have output use another language than the default English for text and speech. See the -ro option and uroman.pl
                        program needed.
  -pd PERSISTDIRECTORY, --persistdirectory PERSISTDIRECTORY
                        Persist directory for Chroma Vector DB used for web page lookups and document analysis.
  -m MODEL, --model MODEL
                        File path to model to load and use. Default is models/zephyr-7b-alpha.Q8_0.gguf
  -em EMBEDDINGMODEL, --embeddingmodel EMBEDDINGMODEL
                        File path to embedding model to load and use. Use a small simple one to keep it fast. Default is
                        models/q4-openllama-platypus-3b.gguf
  -ag, --autogenerate   Keep autogenerating the conversation without interactive prompting.
  -ss, --streamspeak    Speak the text as tts token count chunks.
  -tts TOKENSTOSPEAK, --tokenstospeak TOKENSTOSPEAK
                        When in streamspeak mode, the number of tokens to generate before sending to TTS text to speech.
  -aittss AITTSSEED, --aittsseed AITTSSEED
                        AI Bot TTS 'Seed' to fix the voice models speaking sound instead of varying on input. Set to 0 to allow
                        variance per line spoken.
  -usttss USTTSSEED, --usttsseed USTTSSEED
                        User Bot TTS 'Seed' to fix the voice models speaking sound instead of varying on input. Set to 0 to allow
                        variance per line spoken.
  -mtts MINTOKENSTOSPEAK, --mintokenstospeak MINTOKENSTOSPEAK
                        Minimum number of tokens to generate before sending to TTS text to speech.
  -q QUESTION, --question QUESTION
                        Question to ask initially, else you will be prompted.
  -un USERNAME, --username USERNAME
                        Your preferred name to use for your character.
  -up USERPERSONALITY, --userpersonality USERPERSONALITY
                        Users (Your) personality.
  -ap AIPERSONALITY, --aipersonality AIPERSONALITY
                        AI (Chat Bot) Personality.
  -an AINAME, --ainame AINAME
                        AI Character name to use.
  -asr AISPEAKINGRATE, --aispeakingrate AISPEAKINGRATE
                        AI speaking rate of TTS speaking.
  -ans AINOISESCALE, --ainoisescale AINOISESCALE
                        AI noisescale for TTS speaking.
  -apr AISAMPLINGRATE, --aisamplingrate AISAMPLINGRATE
                        AI sampling rate of TTS speaking, do not change from 16000!
  -usr USERSPEAKINGRATE, --userspeakingrate USERSPEAKINGRATE
                        User speaking rate for TTS.
  -uns USERNOISESCALE, --usernoisescale USERNOISESCALE
                        User noisescale for TTS speaking.
  -upr USERSAMPLINGRATE, --usersamplingrate USERSAMPLINGRATE
                        User sampling rate of TTS speaking, do not change from 16000!
  -sts STOPTOKENS, --stoptokens STOPTOKENS
                        Stop tokens to use, do not change unless you know what you are doing!
  -ctx CONTEXT, --context CONTEXT
                        Model context, default 32768.
  -mt MAXTOKENS, --maxtokens MAXTOKENS
                        Model max tokens to generate, default unlimited or 0.
  -gl GPULAYERS, --gpulayers GPULAYERS
                        GPU Layers to offload model to.
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature to set LLM Model.
  -d, --debug           Debug in a verbose manner.
  -dd, --doubledebug    Extra debugging output, very verbose.
  -s, --silent          Silent mode, No TTS Speaking.
  -ro, --romanize       Romanize LLM output text before input into TTS engine.
  -e, --episode         Episode mode, Output an TV Episode format script.
  -pc PROMPTCOMPLETION, --promptcompletion PROMPTCOMPLETION
                        Prompt completion like... Question: {user_question} Answer:
  -re ROLEENFORCER, --roleenforcer ROLEENFORCER
                        Role enforcer statement with {user} and {assistant} template names replaced by the actual ones in use.
  -sd, --summarizedocs  Summarize the documents retrieved with a summarization model, takes a lot of resources.
  -udb URLSDB, --urlsdb URLSDB
                        SQL Light retrieval URLs DB file location.
  -cdb CHATDB, --chatdb CHATDB
                        SQL Light DB Twitch Chat file location.
  -ectx EMBEDDINGSCONTEXT, --embeddingscontext EMBEDDINGSCONTEXT
                        Embedding Model context, default 512.
  -ews EMBEDDINGWINDOWSIZE, --embeddingwindowsize EMBEDDINGWINDOWSIZE
                        Document embedding window size, default 256.
  -ewo EMBEDDINGWINDOWOVERLAP, --embeddingwindowoverlap EMBEDDINGWINDOWOVERLAP
                        Document embedding window overlap, default 25.
  -eds EMBEDDINGDOCSIZE, --embeddingdocsize EMBEDDINGDOCSIZE
                        Document embedding window overlap, default 4096.
  -hctx HISTORYCONTEXT, --historycontext HISTORYCONTEXT
                        User history context stored and sent to the LLM, default 8192.
  -im IMAGEMODEL, --imagemodel IMAGEMODEL
                        Stable Diffusion Image Model to use.
  -ns, --nosync         Don't sync the text with the speaking, output realtiem.
  -tw, --twitch         Twitch mode, output to twitch chat.
  -gu, --geturls        Get URLs from the prompt and use them to retrieve documents.
  -si, --saveimages     Save images to disk.
  -ll LOGLEVEL, --loglevel LOGLEVEL
                        Logging level: debug, info...
  -ars AUDIOPACKETREADSIZE, --audiopacketreadsize AUDIOPACKETREADSIZE
                        Size of audio packet read/write
  -ren, --render        Render the output to a GUI OpenCV window for playback viewing.
  -wi WIDTH, --width WIDTH
                        Width of rendered window, only used with -ren
  -he HEIGHT, --height HEIGHT
                        Height of rendered window, only used with -ren
  -as, --ascii          Render ascii images
```

## Contributing

Feel free to fork the project, open a PR, or submit issues with suggestions, corrections, or improvements.

For more information, please contact Chris Kennedy at The Groovy Organization.

## License

This project is under the Apache License and is maintained by Chris Kennedy of The Groovy Organization.

## Try it out!

Embark on a journey of interactive conversations, rich auditory experiences, and explore the realms of AI-human interactions with Conscious Chat.
