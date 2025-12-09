import os
from litellm import completion
from NiW.constants import API_BASE_URL, QUERY_FORMULATION_MODEL

SUMMARY_PROMPT = """
You will be given an article. Your task is provide a summary for the article in no more than 7000 tokens."""

CLAIM_EXTRACTION_PROMPT = """ 
You need to extract all the claims from the given article, formulating them as a list of declarative sentences. The claims should be self-contained, so you must avoid using pronouns or relative time references. Only focus on the contents of the article, and ignore the source, author, contributor, or any other information that is not part of the article itself. Only include claims that are clear, factual and verifiable. Do not include anything that is based on your interpretation.
"""

CLAIM_EXTRACTION_EXAMPLES = [
    {"role": "user", "content":\
"""India’s luxury market is on a dramatic ascent, estimated to expand from $7.73 billion in 2023 to $11.3 billion by 2028 — a rate that would likely outpace most of the world’s major luxury markets, according to global consulting firm Kearney.

This projected growth is fueled by a rising middle class, increased urbanization and a new generation of brand-conscious, internationally minded young consumers. But today’s Indian luxury consumer is “no longer a singular archetype,” according to the celebrated Indian fashion designer Gaurav Gupta."""},
    {"role": "assistant", "content":\
"""1. India’s luxury market is experiencing significant and rapid growth.

2. The value of India’s luxury market is projected to increase from $7.73 billion in 2023 to $11.3 billion by 2028.

3. The growth rate of India’s luxury market is expected to surpass that of most other major luxury markets around the world.

4. The consulting firm Kearney is the source of the projection regarding India’s luxury market growth.

5. The anticipated growth of India’s luxury market is driven by a rising middle class, increasing urbanization, and a new generation of young consumers who are brand-conscious and internationally minded.

6. The profile of the Indian luxury consumer has become diverse and can no longer be described by a single archetype.

7. Indian fashion designer Gaurav Gupta has stated that today’s Indian luxury consumer does not fit a singular archetype."""}
]

QUESTION_FORMULATION_PROMPT = """
You will be given an article and a list of claims extracted from it. For each of the claims, you need to mask the central part of it, replacing the central part of it with a generic expression. For each claim, only mask ONE element of it. For different kinds of information you need to mask, you may use `someone` to replace a person's name, `something` to replace a certain thing, `in a certain way` to replace a certain action or process, `in a certain state` to replace some adjectives, etc. Importantly, whenever a piece of information is masked, it should not appear in any of the other masked claims.
"""

QUESTION_FORMULATION_EXAMPLES = [
    {"role": "user", "content": \
"""India’s luxury market is on a dramatic ascent, estimated to expand from $7.73 billion in 2023 to $11.3 billion by 2028 — a rate that would likely outpace most of the world’s major luxury markets, according to global consulting firm Kearney.

This projected growth is fueled by a rising middle class, increased urbanization and a new generation of brand-conscious, internationally minded young consumers. But today’s Indian luxury consumer is “no longer a singular archetype,” according to the celebrated Indian fashion designer Gaurav Gupta. """},
    {"role": "user", "content": \
"""The consulting firm Kearney is the source of the projection regarding India’s luxury market growth.

The anticipated growth of India’s luxury market is driven by a rising middle class, increasing urbanization, and a new generation of young consumers who are brand-conscious and internationally minded.

Indian fashion designer Gaurav Gupta has stated that today’s Indian luxury consumer does not fit a singular archetype."""},
    {"role": "assistant", "content": \
"""Something is the source of the projection regarding the growth of India’s luxury market.

A rising middle class and increasing urbanization contribute to India's market in some way.

Someone has stated that today’s Indian luxury consumer does not fit a singular archetype."""}
]

def summarize_article(article: str):
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": article}
    ]
    create_params = {
        "model": QUERY_FORMULATION_MODEL,
        "messages": messages,
        "stream": False,
        "base_url": API_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
    }
    for i in range(10):
        try:
            raw_response = completion(**create_params)
            break
        except Exception as e:
            print(e)
            continue
    response = raw_response.choices[0].message.content
    return response.strip()

def extract_claims(article: str):
    messages = [{"role": "system", "content": CLAIM_EXTRACTION_PROMPT}]
    messages.extend(CLAIM_EXTRACTION_EXAMPLES)
    messages.append({"role": "user", "content": article})
    create_params = {
        "model": QUERY_FORMULATION_MODEL,
        "messages": messages,
        "stream": False,
        "base_url": API_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
    }
    for i in range(10):
        try:
            raw_response = completion(**create_params)
            response = raw_response.choices[0].message.content
            raw_claims = response.strip().split("\n")
            break
        except Exception as e:
            print(e)
            continue
    # remove all empty entries from claims
    raw_claims = [claim.strip() for claim in raw_claims if claim.strip()]
    claims = []
    for i, claim in enumerate(raw_claims):
        index = claim.find(".")
        if index != -1 and index < 5:
            claim = claim[claim.find(".")+1:].strip()
            claims.append(claim)
    claims = [claim.strip() for claim in claims if claim.strip()]
    return claims

def formulate_questions(article: str, claims: list):
    messages = [{"role": "system", "content": QUESTION_FORMULATION_PROMPT}]
    messages.extend(QUESTION_FORMULATION_EXAMPLES)
    messages.extend([
        {"role": "user", "content": article},
        {"role": "user", "content": "\n".join(claims)}
    ])
    create_params = {
        "model": QUERY_FORMULATION_MODEL,
        "messages": messages,
        "stream": False,
        "base_url": API_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
    }
    for i in range(10):
        try:
            raw_response = completion(**create_params)
            break
        except Exception as e:
            print(e)
            continue
    response = raw_response.choices[0].message.content
    questions = response.strip().split("\n")
    questions = [question.strip() for question in questions if question.strip()]
    return questions

if __name__ == "__main__":
    with open("news_content.md", "r") as f:
        content = f.read()
    summary = summarize_article(content)
    print(f"Summary: {summary}")
    with open("summary.md", "w") as f:
        f.write(summary)
    claims = extract_claims(content)
    with open("claims.md", "w") as f:
        f.write("\n".join(claims))
    print("Extracted Claims:")
    for claim in claims:
        print(f" - {claim}")
    questions = formulate_questions(content, claims)
    print("\nFormulated Questions:")
    for question in questions:
        print(f" - {question}")
    with open("questions.md", "w") as f:
        f.write("\n".join(questions))
    print("\nQuestions saved to questions.md")