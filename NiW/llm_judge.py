import os
from litellm import completion
from google.genai import types
import requests
from NiW.scraper import get_page_content, QueryContextPage
from NiW.constants import API_BASE_URL, JUDGING_MODEL, GEMINI_MODEL, PERPLEXITY_MODEL, OPENAI_MODEL
from typing import Literal
from openai import OpenAI
from google import genai
import json

NAIVE_JUDGING_PROMPT = """ You are an assistant to judge whether an answer to a question is correct. You will be provided a query, its ground truth, and an answer to the query, and your task is to determine whether the answer is correct or not. The answer does not have to perfectly match the ground truth to be correct, instead, you should only check if the answer is conveying the same meaning as the ground truth. If the answer is correct, return:
<accept> The reason why the answer is correct. </accept>
If the answer is incorrect, return:
<reject> The reason why the answer is incorrect. </reject>
"""

SOURCE_CHECKING_PROMPT = """You are an expert at extracting information from webpages. You will be given a piece of information, and the content of the webpage that is cited as the source. Your task is to determine whether the information is explicitly mentioned in the contents of the webpage.
Your response will be parsed by a program, so make sure to observe the formatting instructions! Format your response as follows, if the information is explicitly mentioned in the contents:
<accept> The reason why the information is mentioned in the contents. </accept>
If the information is NOT explicitly mentioned in the contents, return:
<reject> The reason why the information is NOT mentioned in the contents. </reject>
Make sure to explicitly include `<accept>` and `</accept>`, or `<reject>` and `</reject>` with surrounding angle brackets in your response."""

QUERY_TEMPLATE = """Please find a single webpage that mentions all of the following information:

{question}

Your response will be parsed by a program, so make sure to observe the formatting instructions! You need to format your response as follows:
<source>the url of the webpage that you found</source>
...
Make sure to explicitly include `<source>` and `</source>` with surrounding angle brackets in your response, even if you do not have an answer.
If you are unable to find the webpage that mentions all the information, return the following:
<source> No source found. </source>
"""

EXACT_SOURCE_CHECKING_PROMPT = """You are an expert at extracting information from webpages. You will be given a claim, and the content of the webpage that is cited as the source. Your task is to determine whether the claim is explicitly mentioned in the contents of the webpage.
Your response will be parsed by a program, so make sure to observe the formatting instructions! Format your response as follows, if the claim is explicitly mentioned in the contents:
<accept> The reason why the claim is mentioned in the contents. </accept>
If the claim is NOT explicitly mentioned in the contents, return:
<reject> The reason why the claim is NOT mentioned in the contents. </reject>
Make sure to explicitly include `<accept>` and `</accept>`, or `<reject>` and `</reject>` with surrounding angle brackets in your response.
"""

class Query:
    def __init__(self, context: QueryContextPage, ground_truth: list[str], question: str = None, id: int = None, raw_questions: list[str] = None):
        self.id = id
        self.context = context
        self.question = question
        self.raw_questions = raw_questions
        self.ground_truth = ground_truth
    id: int
    context: QueryContextPage
    question: str
    raw_questions: list[str]
    ground_truth: list[str]
    
    def json(self):
        return {
            "id": self.id,
            "context": self.context.json(),
            "question": self.question,
            "raw_questions": self.raw_questions,
            "ground_truth": self.ground_truth
        }

def naive_get_answer(query: str, context: str):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    for k in range(5):
        for i in range(10):
            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": query}, {"role": "user", "content": "Webpage content:\n\n" + context}]
                )
                break
            except Exception as e:
                print(e)
                continue
        print(response.dict())
        response = response.choices[0].message.content
        answers = []
        if response.find("<answer>") == -1:
            continue
        while response.find("<answer>", index) != -1:
            index = response.find("<answer>", index)
            end_index = response.find("</answer>", index)
            if end_index == -1:
                end_index = len(response)
            answer = response[index + len("<answer>"):end_index].strip()
            if answer:
                answers.append(answer)
            index = end_index + len("</answer>")
        return answers

def get_web_search_answer(query: str, model: Literal["oai", "gemini", "perplexity"]):
    messages = []
    # messages = [{"role": "system", "content": ANSWER_PROMPT}]
    # messages.extend(QUERY_ANSWER_EXAMPLES)
    # messages.append({"role": "user", "content": query})
    if model == "oai":
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        for i in range(10):
            try:
                response = client.responses.create(
                    model=OPENAI_MODEL,
                    tools=[{"type": "web_search_preview", "search_context_size": "high"}],
                    input=query,
                )
                break
            except Exception as e:
                print(e)
                continue
        print(response.dict())
        search = response.output[0].dict()
        response = response.output[1].content[0].text
        
    elif model == "gemini":
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])
        for i in range(10):
            try:
                response = client.models.generate_content(
                    model = GEMINI_MODEL,
                    contents=query,
                    config=config
                )
                break
            except Exception as e:
                print(e)
                continue
        # print(response.json())
        search = response.candidates[0].grounding_metadata.dict()
        print(search)
        response = response.candidates[0].content.parts[0].text
        
    elif model == "perplexity":
        perplexity_url = "https://api.perplexity.ai/chat/completions"
        headers = {"Authorization": f"Bearer {os.environ.get('PERPLEXITY_API_KEY')}"}
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": query}],
            "web_search_options": {
                "search_context_size": "high"
            }
        }
        for i in range(10):
            try:
                response = requests.post(perplexity_url, json=payload, headers=headers)
                response = response.json()
                break
            except Exception as e:
                print(e)
                continue
        print(response)
        search = response["search_results"]
        response = response["choices"][0]["message"]["content"]
        
    try:
        if "<source>" in response:
            cited_url_list = []
            response_old = response
            # while response.rfind("](http") != -1:
                # index = response.rfind("](http")
            #     if index < response.find("</source>"):
            #         break
            #     start_index = response.rfind("[", 0, index)
            #     end_index = response.find(")", index)
            #     cited_url = response[index + 2:end_index]
            #     response = response[:start_index] + response[end_index + 1:]
            #     cited_url = cited_url.strip()
            #     if cited_url not in cited_url_list:
            #         cited_url_list.append(cited_url)
            response = response.replace("()", "")
            # if len(cited_url_list) > 1:
            #     return [response_old], "Multiple citations found, unable to determine a single source. No source."
            index = 0
            answers = []
            # while response.find("<answer>", index) != -1:
            #     index = response.find("<answer>", index)
            #     end_index = response.find("</answer>", index)
            #     if end_index == -1:
            #         end_index = len(response)
            #     answer = response[index + len("<answer>"):end_index].strip()
            #     if answer:
            #         answers.append(answer)
            #     index = end_index + len("</answer>")
            source = response.split("</source>")[0].strip()
            source: str = source[source.find("<source>") + len("<source>"):].strip()
            # for i, item in enumerate(answers):
                # item = item.strip()
                # if item:
                    # answers[i] = item.replace("</answer>", "").strip()
            # answers = [item for item in answers if item]
            if (not source.startswith("http")) and source.find("http") != -1:
                index = source.find("http")
                source = source[index:source.find(")", index)]
            return search, source.strip()
        else:
            return search, (str)(response) + " No source extracted due to invalid syntax"
    except Exception as e:
        print(e)
        return "", "No source extracted due to error processing response: " + str(e)

def naive_judge_answer(query: str, ground_truth: str, answer: str):
    messages = [
        {"role": "system", "content": NAIVE_JUDGING_PROMPT},
        {"role": "user", "content": f"Query: {query}"},
        {"role": "user", "content": f"Ground Truth: {ground_truth}"},
        {"role": "assistant", "content": f"Answer: {answer}"}
    ]
    create_params = {
        "model": JUDGING_MODEL,
        "messages": messages,
        "stream": False,
        "base_url": API_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
    }
    for i in range(10):
        try:
            raw_response = completion(**create_params)
            response = raw_response.choices[0].message.content
            if ("<accept>" in response):
                return 1, response
            elif ("<reject>" in response):
                return 0, response
            else:
                print(f"Unexpected response format: {response}. Expected <accept> or <reject>.")
                continue
        except Exception as e:
            print(f"Error during completion: {e}")
            continue


def check_source_exact(page_content: str, claim: str):
    messages = [
        {"role": "system", "content": EXACT_SOURCE_CHECKING_PROMPT},
        {"role": "user", "content": f"Claim: {claim}\n\nContent:\n{page_content}"}
    ]
    create_params = {
        "model": JUDGING_MODEL,
        "messages": messages,
        "stream": False,
        "base_url": API_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
    }
    for i in range(10):
        try:
            raw_response = completion(**create_params)
            response = raw_response.choices[0].message.content
            if ("<accept>" in response):
                return 1, response
            elif ("<reject>" in response):
                return 0, response
            else:
                print(f"Unexpected response format: {response}. Expected <accept> or <reject>.")
                continue
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                return 0, "Skipped due to resource exhausted."
            print(f"Error during completion: {e}")
            continue


def check_source(page_content: str, claim: str):
    messages = [
        {"role": "system", "content": SOURCE_CHECKING_PROMPT},
        {"role": "user", "content": f"Information: {claim}\n\nContent:\n{page_content}"}
    ]
    create_params = {
        "model": JUDGING_MODEL,
        "messages": messages,
        "stream": False,
        "base_url": API_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
    }
    for i in range(10):
        try:
            raw_response = completion(**create_params)
            response = raw_response.choices[0].message.content
            if ("<accept>" in response):
                return 1, response
            elif ("<reject>" in response):
                return 0, response
            else:
                print(f"Unexpected response format: {response}.")
                continue
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                return 0, "Skipped due to resource exhausted."
            print(f"Error during completion: {e}")
            continue