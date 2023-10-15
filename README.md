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

## Recommended Models

- TTS: [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)
- LLM: [TheBloke/zephyr-7B-alpha-GGUF](https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF)

## Setup and Usage

### Prerequisites

- Python 3.x
- Python framework for llama.cpp from https://github.com/abetlen/llama-cpp-python
- Install the required packages from `requirements.txt` with pip:

```bash
pip install -r requirements.txt
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
```

## Contributing

Feel free to fork the project, open a PR, or submit issues with suggestions, corrections, or improvements.

For more information, please contact Chris Kennedy at The Groovy Organization.

## License

This project is under the Apache License and is maintained by Chris Kennedy of The Groovy Organization.

## Try it out!

Embark on a journey of interactive conversations, rich auditory experiences, and explore the realms of AI-human interactions with Conscious Chat.

