```
Chatbot that speaks

Multispeaker LLM locally with TTS all free if run locally

Chris Kennedy (C) The Groovy Organization Apache License
```

# Conscious Chat: Multispeaker LLM with TTS

Conscious Chat is an innovative project aimed at emulating conscious interactions using multispeaker Language Models (LLM) alongside Text-to-Speech (TTS) technology. This setup facilitates visual and auditory cues, creating a more engaging and interactive environment for users. The end goal is to foster multi-user interactions among multiple AI entities, creating a Game of Life-like phenomenon for human entertainment and question answering.

## Features

- **Multispeaker LLM**: Engage in conversation with AI entities possessing unique personalities.
- **Text-to-Speech (TTS)**: Hear responses audibly, adding a layer of realism to the interaction.
- **Customizable Personalities**: Tailor the personality of AI and user entities to your liking.
- **Local Execution**: All functionalities are available for free when run locally.

## Recommended models

- TTS: https://huggingface.co/facebook/mms-tts-eng
- LLM: https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF

## Setup and Usage

### Prerequisites

- Python 3.x
- Python framework for llama.cpp from https://github.com/abetlen/llama-cpp-python
- Install the required packages from `requirements.txt` using pip:
  ```bash
  pip install -r requirements.txt
  ```

### Execution

Run the `characterChat.py` script from your terminal to initiate a chat session:

```bash
./characterChat.py [options]
```

#### Options:

```plaintext
  -h, --help                        Show this help message and exit
  -m, --model MODEL                 Path to the model
  -q, --question QUESTION           Question to ask the AI
  -un, --username USERNAME          Username
  -up, --userpersonality USERPERSONALITY     User Personality description
  -ap, --aipersonality AIPERSONALITY         AI Personality description
  -an, --ainame AINAME              AI Name
  -mt, --maxtokens MAXTOKENS        Max tokens for response
  -asr, --aispeakingrate AISPEAKINGRATE      AI Speaking rate
  -ans, --ainoisescale AINOISESCALE         AI Noise scale
  -apr, --aisamplingrate AISAMPLINGRATE      AI Sampling rate
  -usr, --userspeakingrate USERSPEAKINGRATE  User Speaking rate
  -uns, --usernoisescale USERNOISESCALE      User Noise scale
  -upr, --usersamplingrate USERSAMPLINGRATE  User Sampling rate
```

## License

This project is under the Apache License and is maintained by Chris Kennedy of The Groovy Organization.

## Contributing

Feel free to fork the project, open a PR, or submit issues with suggestions, corrections, or improvements.

---

Embark on a journey of interactive conversations, rich auditory experiences, and explore the realms of AI-human interactions with Conscious Chat.

---

This organized and refined README provides a clear overview of the project, its features, how to set it up, and how to use it, while also maintaining a professional and inviting tone for potential contributors or users.
