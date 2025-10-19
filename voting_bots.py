"""Voting Bots

Run three LLMs in parallel, have each vote on the others' answers, and return the majority-selected response.

Usage:
    python3 voting_bots.py  # Run with stub clients
    
    # Or in your code:
    from voting_bots import get_recommendations, get_votes
"""

import os
import json
import re
import requests
import gradio as gr
from bs4 import BeautifulSoup
from collections import Counter
from dotenv import load_dotenv
from typing import Any, Dict, List
from openai import OpenAI
import anthropic
import google.generativeai
from utils import(
    clean_str,
    filter_response,
    is_valid_response,
    clean_prompt,
)

# Constants for model names
MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = "llama3.2"
MODEL_GEMINI = "gemini-2.5-flash-lite"
MODEL_CLAUDE = "claude-4.0-sonnet"

## API KEY Validation(printing key prefix to help with debugging)
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
claude_api_key = os.getenv('CLAUDE_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
if claude_api_key:
    print(f"Claude API Key exists and begins {claude_api_key[:7]}")
else:
    print("Claude API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

#Connect to LLMS
openai = OpenAI()

claude = anthropic.Anthropic()

google.generativeai.configure()

#Not setting a system prompt as our usecase is general and not specific to any domain.




# --------------------------- Pluggable clients (stubs) ---------------------------
# In the notebook CLIENTS and MODELS map LLM names to client objects and model ids.
# For extraction we provide minimal stubs that reproduce the notebook behavior shape.

class StubClient:
    def __init__(self, name: str, response_map: Dict[str, str]):
        self.name = name
        self._responses = response_map

    def chat_completion(self, model: str, messages: List[Dict[str, str]]):
        # Return a simple object with a 'choices' list like the notebook expects for non-Anthropic
        key = self.name
        content = self._responses.get(key, '{}')
        return type('R', (), {'choices': [{'message': type('M', (), {'content': content})}]})

    # compatibility wrapper used in our adapter
    def chat(self):
        class C:
            def __init__(self, parent):
                self.parent = parent

            def completions_create(self, model, messages, response_format=None, max_tokens=None):
                return self.parent.chat_completion(model=model, messages=messages)

        return C(self)


# Simple anthopic/claude stub with a different signature
class StubAnthropicClient(StubClient):
    class _Resp:
        def __init__(self, text):
            self.content = [type('T', (), {'text': text})]

    class messages:
        @staticmethod
        def create(model, max_tokens, system, messages):
            # not used in demo
            return StubAnthropicClient._Resp('{}')


# Default MODELS/CLIENTS used by the module (can be replaced by user)
MODEL_GPT = "{'GPT': 'gpt-4o-mini', 'CLAUDE': 'claude-3', 'GEMINI': 'gemini-2'}"
CLIENTS = {
    'GPT': StubClient('GPT', response_map={
        'GPT': json.dumps({"Optimized Title": "GPT Title", "Justification": "GPT justification"})
    }),
    'GEMINI': StubClient('GEMINI', response_map={
        'GEMINI': json.dumps({"Optimized Title": "Gemini Title", "Justification": "Gemini justification"})
    }),
    'CLAUDE': StubAnthropicClient('CLAUDE', response_map={
        'CLAUDE': json.dumps({"Optimized Title": "Claude Title", "Justification": "Claude justification"})
    }),
}


# --------------------------- Core functions (extracted) ---------------------------

def get_model_response(model: str, messages: List[Dict[str, str]]):
    """Adapter that calls the client for the given model and returns the textual content.

    Notes: The original notebook had special-casing for CLAUDE; we mimic that at high level.
    """
    if model == 'CLAUDE':
        # Anthropic-style response
        client = CLIENTS[model]
        # original used client.messages.create with system/messages separated
        # We'll just return the stubbed text in demo
        return client._responses.get(model, '{}')
    else:
        client = CLIENTS[model]
        # notebook called client.chat.completions.create(...)
        # our stub exposes chat().completions_create
        resp = client.chat().completions_create(model=MODELS[model], messages=messages)
        return resp.choices[0].message.content


def get_title(model: str, article: str):
    messages = [
        {"role": "system", "content": clean_prompt("SEO writer system prompt")},
        {"role": "user", "content": f"Suggest one optimized title for the following article: {article}"},
    ]

    while True:
        response = get_model_response(model=model, messages=messages)
        response = filter_response(response)
        response = clean_str(response)
        try:
            response_obj = json.loads(response)
        except Exception:
            # fallback: return a minimal constructed object
            response_obj = {"Optimized Title": f"{model} fallback title", "Justification": "fallback"}

        required_keys = {"Optimized Title", "Justification"}
        cleaned, ok = is_valid_response(original_dict=response_obj, required_keys=required_keys)
        if ok:
            cleaned["Author"] = model
            return cleaned
        # in the original, this loop would retry until valid; for the script, break to avoid infinite loop
        return {"Optimized Title": f"{model} title (unvalidated)", "Justification": "(unvalidated)", "Author": model}


def get_recommendations(article: str):
    models = ['GEMINI', 'CLAUDE', 'GPT']
    recommendations = []
    with ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(get_title, model, article): model for model in models}
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                recommendations.append(result)
            except Exception as e:
                print(f"Error getting title from {model}: {e}")
    return {'recommendations': recommendations}


# Voting pieces

def get_model_vote(arguments: Dict[str, Any]):
    model = arguments['model']
    recommendations = arguments['recommendations']

    vote_system_prompt = """
        I'm sending you a list of suggested titles for an article, their justification, and the authors suggesting each title.
        Select which title you think is the best based on the justifications.
        Please respond in valid JSON format only. 
        Response Format: {"vote": [insert here the title you selected as the best]}
    """

    vote_user_prompt = "Which of the suggested titles do you think is the best for the article?"

    messages = [
        {"role": "system", "content": vote_system_prompt},
        {"role": "user", "content": f"{vote_user_prompt} {recommendations}"},
    ]

    while True:
        response = get_model_response(model=model, messages=messages)
        response = filter_response(response)
        if response:
            try:
                response_obj = json.loads(response)
            except Exception:
                # fallback: pick first recommendation
                response_obj = {"vote": recommendations[0]['Optimized Title'] if recommendations else None}

            required_keys = {"vote"}
            cleaned, ok = is_valid_response(original_dict=response_obj, required_keys=required_keys)
            if ok:
                cleaned['voter'] = model
                return cleaned
            # fallback return
            return {'vote': recommendations[0]['Optimized Title'] if recommendations else None, 'voter': model}


def get_votes(recommendations: List[Dict[str, Any]]):
    model_args = [
        {'model': 'GEMINI', 'recommendations': recommendations},
        {'model': 'CLAUDE', 'recommendations': recommendations},
        {'model': 'GPT', 'recommendations': recommendations},
    ]

    votes = []
    with ThreadPoolExecutor() as executor:
        future_to_model = {executor.submit(get_model_vote, args): args['model'] for args in model_args}
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                result = future.result()
                votes.append(result)
            except Exception as e:
                print(f"Error getting vote from {model}: {e}")

    winner = get_winner(votes)
    return {'votes': votes, 'winner': winner}


def get_winner(votes: List[Dict[str, Any]]):
    vote_choices = [v['vote'] for v in votes if 'vote' in v]
    vote_counts = Counter(vote_choices)
    most_common = vote_counts.most_common()
    if len(most_common) == 0:
        return 'No votes were cast.'
    elif len(most_common) == 1 or most_common[0][1] > (most_common[1][1] if len(most_common) > 1 else 0):
        return {'title': most_common[0][0], 'count': most_common[0][1]}
    else:
        return 'There is no clear winner due to a tie.'


# --------------------------- Demo / CLI ---------------------------

def run_demo():
    article = 'This is a short article about testing the Voting Bots extraction.'
    recs = get_recommendations(article)
    print('Recommendations:')
    print(json.dumps(recs, indent=2))

    votes_outcome = get_votes(recs['recommendations'])
    print('\nVotes outcome:')
    print(json.dumps(votes_outcome, indent=2))


if __name__ == '__main__':
    run_demo()
