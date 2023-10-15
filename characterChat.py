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
from llama_cpp import ChatCompletionMessage
import sounddevice as sd
import pyaudio
import wave
import os
import queue
import subprocess
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from bs4 import BeautifulSoup as Soup
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from transformers import logging
logging.set_verbosity_error()
import logging as logger
logger.basicConfig(level=logger.ERROR)
import re
from transformers import pipeline

# Initialize summarization pipeline
summarizer = pipeline("summarization")


def summarize_documents(documents):
    """
    Summarizes the page content of a list of Document objects.

    Parameters:
        documents (list): A list of Document objects.

    Returns:
        str: Formatted string containing document details with summarized content.
    """
    output = []
    for doc in documents:
        source = doc.metadata.get('source', 'N/A')
        title = doc.metadata.get('title', 'N/A')

        # Summarize page content
        summary = summarizer(doc.page_content, max_length=200, min_length=20, do_sample=False)
        summarized_content = summary[0]['summary_text'].strip()

        # Format the extracted and summarized data
        formatted_data = f"Main Source: {source}\nTitle: {title}\nSummarized Content: {summarized_content}\n"
        output.append(formatted_data)

    # Combine all formatted data
    return "\n".join(output)


def parse_documents(documents):
    """
    Parses a list of Document objects and formats the output.

    Parameters:
        documents (list): A list of Document objects.

    Returns:
        str: Formatted string containing document details.
    """
    output = []
    for doc in documents:
        # Extract metadata and page content
        source = doc.metadata.get('source', 'N/A')
        title = doc.metadata.get('title', 'N/A')
        page_content = doc.page_content[:200]  # Get up to N characters

        # Format the extracted data
        formatted_data = f"Main Source: {source}\nTitle: {title}\nDocument Page Content: {page_content}\n"
        output.append(formatted_data)

    # Combine all formatted data
    return "\n".join(output)


def extract_urls(text):
    """
    Extracts all URLs that start with 'http' or 'https' from a given text.

    Parameters:
        text (str): The text from which URLs are to be extracted.

    Returns:
        list: A list of extracted URLs.
    """
    url_regex = re.compile(
        r'http[s]?://'  # http:// or https://
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'  # domain
    )
    return re.findall(url_regex, text)

def gethttp(url, question, llama_embeddings, persistdirectory):
    if url == "" or url == None:
        print("URL is empty for gethttp()")
        return []
    if question == "":
        print("Question is empty for gethttp()")
        return []
    try:
        loader = RecursiveUrlLoader(url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text)
    except Exception as e:
        print("Error with url {url} gethttp WebBaseLoader:", e)
        return []

    docs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            data = loader.load() # Overlap chunks for better context
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            all_splits = text_splitter.split_documents(data)
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=llama_embeddings, persist_directory=persistdirectory)
            docs = vectorstore.similarity_search(question)
        except Exception as e:
            print("Error with {url} text splitting in gethttp():", e)
    return docs

def uromanize(input_string, uroman_path):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    script_path = "uroman.pl"
    if uroman_path != "":
        script_path = os.path.join(uroman_path, "uroman.pl")

    command = ["perl", script_path]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Execute the perl command
    stdout, stderr = process.communicate(input=input_string.encode())

    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

    # Return the output as a string and skip the new-line character at the end
    return stdout.decode()[:-1]

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

def clean_text_for_tts(text):
    # Convert numbers to words
    p = inflect.engine()
    text = re.sub(r'\b\d+(\.\d+)?\b', lambda match: p.number_to_words(match.group()), text)

    # Strip out non-speaking characters
    """
    if args.language == "":
        text = re.sub(r'[^a-zA-Z0-9 .,?!]', '', text)
    """

    # Add a pause after punctuation
    text = text.replace('.', '. ')
    text = text.replace(',', ', ')
    text = text.replace('?', '? ')
    text = text.replace('!', '! ')

    return text

def check_min(value):
    ivalue = int(value)
    if ivalue < 2:
        raise argparse.ArgumentTypeError("%s is an invalid value for the number of tokens to speak! It should be 2 or more." % value)
    return ivalue

default_ai_name = "Usagi"
default_human_name = "AnimeFan"

facebook_model = "facebook/mms-tts-eng"

## AI TTS Model for Speech
aimodel = VitsModel.from_pretrained(facebook_model)
aitokenizer = AutoTokenizer.from_pretrained(facebook_model, is_uroman=True, normalize=True)

## User TTS Model for Speech
usermodel = VitsModel.from_pretrained(facebook_model)
usertokenizer = AutoTokenizer.from_pretrained(facebook_model, is_uroman=True, normalize=True)


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--language", type=str, default="", help="Have output use another language than the default English for text and speech. See the -ro option and uroman.pl program needed.")
parser.add_argument("-pd", "--persistdirectory", type=str, default="vectordb_data",
                    help="Persist directory for Chroma Vector DB used for web page lookups and document analysis.")
parser.add_argument("-m", "--model", type=str, default="/Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-alpha.Q8_0.gguf",
                    help="File path to model to load and use.")
parser.add_argument("-em", "--embeddingmodel", type=str, default="/Volumes/BrahmaSSD/LLM/models/GGUF/zephyr-7b-alpha.Q2_K.gguf",
                    help="File path to embedding model to load and use. Use a small simple one to keep it fast.")
parser.add_argument("-ag", "--autogenerate", type=bool, default=False, help="Keep autogenerating the conversation without interactive prompting.")
parser.add_argument("-ss", "--streamspeak", type=bool, default=False, help="Speak the text as tts token count chunks.")
parser.add_argument("-tts", "--tokenstospeak", type=check_min, default=50, help="When in streamspeak mode, the number of tokens to generate before sending to TTS text to speech.")
parser.add_argument("-ttss", "--ttsseed", type=int, default=1,
                    help="TTS 'Seed' to fix the voice models speaking sound instead of varying on input. Set to 0 to allow variance per line spoken.")
parser.add_argument("-mtts", "--mintokenstospeak", type=check_min, default=12, help="Minimum number of tokens to generate before sending to TTS text to speech.")
parser.add_argument("-q", "--question", type=str, default="", help="Question to ask initially, else you will be prompted.")
parser.add_argument("-un", "--username", type=str, default=default_human_name, help="Your preferred name to use for your character.")
parser.add_argument("-up", "--userpersonality", type=str,
                    default="A magical girl from an anime that is here to talk to other magical girls", help="Your personality.")
parser.add_argument("-ap", "--aipersonality", type=str,
                    default="A magical girl from an anime that is here to help however needed.", help="AI Personality.")
parser.add_argument("-an", "--ainame", type=str, default=default_ai_name, help="AI Character name to use.")
parser.add_argument("-asr", "--aispeakingrate", type=float, default=1.0, help="AI speaking rate of TTS speaking.")
parser.add_argument("-ans", "--ainoisescale", type=float, default=0.667, help="AI noisescale for TTS speaking.")
parser.add_argument("-apr", "--aisamplingrate", type=int,
                    default=aimodel.config.sampling_rate, help="AI sampling rate of TTS speaking, do not change from 16000!")
parser.add_argument("-usr", "--userspeakingrate", type=float, default=0.8, help="User speaking rate for TTS.")
parser.add_argument("-uns", "--usernoisescale", type=float, default=0.667, help="User noisescale for TTS speaking.")
parser.add_argument("-upr", "--usersamplingrate", type=int, default=usermodel.config.sampling_rate,
                    help="User sampling rate of TTS speaking, do not change from 16000!")
parser.add_argument("-sts", "--stoptokens", type=str, default="Question:,%s:,Human:,Plotline:" % (default_human_name),
                    help="Stop tokens to use, do not change unless you know what you are doing!")
parser.add_argument("-ctx", "--context", type=int, default=32768, help="Model context, default 32768.")
parser.add_argument("-ectx", "--embeddingscontext", type=int, default=1024, help="Embedding Model context, default 1024.")
parser.add_argument("-mt", "--maxtokens", type=int, default=0, help="Model max tokens to generate, default unlimited or 0.")
parser.add_argument("-gl", "--gpulayers", type=int, default=0, help="GPU Layers to offload model to.")
parser.add_argument("-t", "--temperature", type=float, default=0.9, help="Temperature to set LLM Model.")
parser.add_argument("-d", "--debug", type=bool, default=False, help="Debug in a verbose manner.")
parser.add_argument("-dd", "--doubledebug", type=bool, default=False, help="Extra debugging output, very verbose.")
parser.add_argument("-s", "--silent", type=bool, default=False, help="Silent mode, No TTS Speaking.")
parser.add_argument("-ro", "--romanize", type=bool, default=False, help="Romanize LLM output text before input into TTS engine.")
parser.add_argument("-e", "--episode", type=bool, default=False, help="Episode mode, Output an TV Episode format script.")
parser.add_argument("-pc", "--promptcompletion", type=str, default="\nQuestion: {user_question}\nAnswer:",
                    help="Prompt completion like...\n\nQuestion: {user_question}\nAnswer:")
parser.add_argument("-re", "--roleenforcer",
                    type=str, default="\nAnswer the question asked by {user}. Stay in the role of {assistant}, give your thoughts and opinions as asked.\n",
                    help="Role enforcer statement with {user} and {assistant} template names replaced by the actual ones in use.")
parser.add_argument("-sd", "--summarizedocs", type=bool, default=False, help="Summarize the documents retrieved with a summarization model, takes a lot of resources")
args = parser.parse_args()

if args.ttsseed > 0:
    set_seed(args.ttsseed)

if args.autogenerate:
    args.stoptokens = ""

if args.doubledebug:
    args.debug = True

if args.episode:
    args.roleenforcer = "%s Format the output like a TV episode script using markdown.\n" % args.roleenforcer
    args.roleenforcer.replace('Answer the question asked by', 'Create a story from the plotline given by')
    args.promptcompletion.replace('Answer:', 'Episode in Markdown Format:')
    args.promptcompletion.replace('Question', 'Plotline')

if args.language != "":
    args.promptcompletion = "%s Speak in the %s language" % (args.promptcompletion, args.language)

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

## LLM Model for Text
llm = Llama(model_path=args.model, n_ctx=args.context, verbose=args.doubledebug, n_gpu_layers=args.gpulayers)
llama_embeddings = LlamaCppEmbeddings(model_path=args.embeddingmodel, n_ctx=args.embeddingscontext, verbose=args.doubledebug, n_gpu_layers=args.gpulayers)

## Human User prompt
def get_user_input():
    if args.episode:
        return input("Plotline: ")
    else:
        return input("Question: ")

## Speak a line
def speak_line(line):
    if args.silent:
        return
    if not line:
        return
    if args.debug:
        print("\n--- Speaking line with TTS...\n")

    ## Numbers to Words
    aitext = convert_numbers_to_words(line)
    ## Romanize
    romanized_aitext = ""
    try:
        uroman_path = ""
        if "UROMAN" in os.environ:
            uroman_path = os.environ["UROMAN"]
        if args.romanize:
            romanized_aitext = uromanize(aitext, uroman_path=uroman_path)
            aitext = romanized_aitext
            if args.debug:
                print("\n--- Romanized Text: %s" % romanized_aitext)
    except Exception as e:
        print("\n--- Error romanizing input:", e)

    ## Tokenize
    aiinputs = aitokenizer(aitext, return_tensors="pt")
    aiinputs['input_ids'] = aiinputs['input_ids'].long()

    ## Run TTS Model
    try:
        with torch.no_grad():
            aioutput = aimodel(**aiinputs).waveform
    except Exception as e:
        print("\n--- Error with TTS AI Speech model!", e)
        return

    ## Buffer audio speech output as WAV
    waveform_np = aioutput.squeeze().numpy().T
    buf = io.BytesIO()
    sf.write(buf, waveform_np, ai_sampling_rate, format='WAV')
    buf.seek(0)

    ## Speak TTS Output
    wave_obj = wave.open(buf)
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wave_obj.getsampwidth()),
                    channels=wave_obj.getnchannels(),
                    rate=wave_obj.getframerate(),
                    output=True)

    ## Read and Speak
    while True:
        data = wave_obj.readframes(1024)
        if not data:
            break
        stream.write(data)

    ## Stop and cleanup speaking TODO keep this open
    stream.stop_stream()
    stream.close()
    p.terminate()

## AI Conversation
def converse(question, messages):
    output = llm.create_chat_completion(
        messages,
        max_tokens=args.maxtokens,
        temperature=args.temperature,
        stream=True,
        stop=args.stoptokens.split(',') if args.stoptokens else []  # use split() result if stoptokens is not empty
    )

    tokens = []
    speaktokens = []
    if args.streamspeak:
        speaktokens.extend([' ', '\n', '.', '?', ','])
    else:
        speaktokens.extend(['\n', '.', '?', ','])

    token_count = 0
    tokens_to_speak = 0
    role = ""
    for item in output:
        if args.doubledebug:
            print("--- Got Item: %s\n" % json.dumps(item))

        delta = item["choices"][0]['delta']
        token = ""
        if 'role' in delta:
            if args.debug:
                print(f"\n--- Found Role: {delta['role']}: ")
            role = delta['role']

        # Check if we got a token
        if 'content' not in delta:
            if args.doubledebug:
                print(f"--- Skipping lack of content: {delta}")
            continue

        token = delta['content']
        sub_tokens = re.split('([ ,.\n?])', token)
        if len(sub_tokens) > 0:
            for sub_token in sub_tokens:
                if sub_token:  # check if sub_token is not empty
                    tokens.append(sub_token)
                    token_count += 1
                    tokens_to_speak += 1
                    print("%s" % sub_token, end='', flush=True)

                    if ((args.streamspeak and tokens_to_speak > args.tokenstospeak) or
                            (tokens_to_speak > args.mintokenstospeak)) and (sub_token[len(sub_token)-1] in speaktokens):
                        line = ''.join(tokens)
                        tokens_to_speak = 0
                        if line.strip():  # check if line is not empty
                            spoken_line = clean_text_for_tts(line)  # clean up the text for TTS
                            speak_line(spoken_line.strip())
                        tokens = []
        else:
            tokens.append(sub_token)
            token_count += 1
            tokens_to_speak += 1
            print("%s" % sub_token, end='', flush=True)

    if tokens:  # if there are any remaining tokens, speak the last line
        line = ''.join(tokens)
        if line.strip():  # check if line is not empty
            spoken_line = clean_text_for_tts(line)  # clean up the text for TTS
            try:
                speak_line(spoken_line.strip())
            except Exception as e:
                print("\n--- Error speaking line!!!:", e)

    return ''.join(tokens).strip()


## Main
if __name__ == "__main__":
    # TODO Create a queue for lines to be spoken
    speak_queue = queue.Queue()

    ## System role
    messages=[
        ChatCompletionMessage(
            # role="user",
            role="system",
            content="You are %s who is %s." % (
                args.ainame,
                args.aipersonality),
        )
    ]

    initial_question = args.question

    ## Question
    if (initial_question != ""):
        prompt = "%s: You are %s\n%s%s:" % (
                args.ainame,
                args.aipersonality,
                args.roleenforcer.replace('{user}', args.username).replace('{assistant}', args.ainame),
                args.promptcompletion.replace('{user_question}', initial_question))

        ## User Question
        messages.append(ChatCompletionMessage(
                role="user",
                content="%s" % initial_question,
            ))

        print("%s" % prompt)
        question_spoken = clean_text_for_tts(initial_question)
        speak_line(question_spoken)

        response = converse(initial_question, messages)

        ## AI Response
        messages.append(ChatCompletionMessage(
                role="assistant",
                content="%s" % response,
            ))

    while True:
        try:
            print("\n\n--- You can press the <Return> key for the output to continue where it last left off.")
            if args.episode:
                print("--- Create your plotline ", end='');
            else:
                print("--- Ask your question ", end='');
            print("and press the Return key to continue, or Ctrl+C to exit the program.\n")

            next_question = get_user_input()
            urls = extract_urls(next_question)
            context = "Context:\n"
            try:
                for url in urls:
                    docs = gethttp(url, next_question, llama_embeddings, args.persistdirectory)
                    if args.debug:
                        print("--- GetHTTP found {url} with %d docs" % len(docs))
                    if len(docs) > 0:
                        if args.summarizedocs:
                            parsed_output = summarize_documents(docs) # parse_documents gets more information with less precision
                        else:
                            parsed_output = parse_documents(docs)
                        context = "%s\n%s\n\n" % (context, parsed_output.strip())
            except Exception as e:
                print("Error with url retrieval:", e)

            prompt = "%sUse the above Context if any exists to help form the output. You are %s who is %s Using the Context as inspiration and references.\n%s%s" % (
                    context,
                    args.ainame,
                    args.aipersonality,
                    args.roleenforcer.replace('{user}', args.username).replace('{assistant}', args.ainame),
                    args.promptcompletion.replace('{user_question}', next_question))

            if args.debug:
                print("\n--- Using Prompt:\n---\n%s\n---\n" % prompt)

            ## User Question
            messages.append(ChatCompletionMessage(
                    role="user",
                    content="%s" % prompt,
                ))

            # Generate the Answer
            if args.episode:
                print("\n--- Generating an episode from your plotline...", end='');
            else:
                print("\n--- Generating the answer to your question...", end='');
            print ("   (this may take awhile without a big GPU)")
            response = converse(next_question, messages)

            ## AI Response History
            messages.append(ChatCompletionMessage(
                    role="assistant",
                    content="%s" % response,
                ))
        except KeyboardInterrupt:
            print("\n--- Exiting...")
            break

