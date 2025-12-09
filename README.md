# Needle in the Web

This repository implements the **Needle in the Web** benchmark, a comprehensive evaluation framework for assessing large language models' (LLMs) capabilities in web-based information retrieval and verification.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tango-Whiskyman/Needle_in_the_Web.git
cd Needle_in_the_Web
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys as environment variables:
```bash
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"
export PERPLEXITY_API_KEY="your_perplexity_key"
export FIRECRAWL_API_KEY="your_firecrawl_key"
```

## Quick Start

To run the full evaluation pipeline on an existing queryset:

```bash
python -m main your_experiment_id all gemini
```

Replace `your_experiment_id` with your desired experiment name, `path/to/web_content.json` with the path to the queryset, `all` with the difficulty mode (`easy`, `medium`, `hard`, or `all`), and `gemini` with your chosen model (`oai`, `gemini`, or `perplexity`).

For running on existing querysets, see the detailed instructions below.

## How to Reproduce Results

### Running Existing Experiments

1. **Evaluate on Existing Querysets**: Evaluate a model on existing querysets.

```python
from main import *

full_pipeline(
    experiment_id="name_of_the_experiment",
    model="gemini",  # "oai", "gemini", "perplexity"
    queryset_specs=[{"name": "name_of_the_queryset", "filename": "path_to_the_queryset"}, {"name": "name_of_another_queryset", "filename": "path_to_another_queryset"}] # name of the queryset is only used in naming result files and logs.
)
```

2. **Only Collect Model Answers**: Only collect a model's answers on an existing queryset without judging them.

```python
from main import *

get_model_answer(
	experiment_id="name_of_the_experiment",
	model="gemini", # "oai", "gemini", "perplexity"
	qyertset_name="name_of_the_queryset"
	queryset_path="path_to_the_queryset"
)
```

3. **Judging Model Answers**: Judging a set of pre-collected answers.

```python
from main import *

judge_model_answer(
	experiment_id="name_of_the_experiment",
	raw_results_path="path_to_the_results",
	queryset_path="path_to_the_querysets",
	queryset_name="name_of_the_queryset"
)
```

### Understanding Results

Results are saved in `experiments/{experiment_id}/test_results/` with metrics including:
- **Accuracy**: Percentage of queries where a correct source was found
- **Ground Truth Match**: Queries that found the exact original webpage
- **Criteria Match**: Queries that found alternative correct sources
- **Invalid Source**: Queries with no valid source provided
- **Wrong Webpage**: Queries with incorrect sources

## Creating New Querysets

### Step 1: Collect Web Content

First, gather web content from your target domain. You can use the built-in scrapers, which requires a FireCrawl api key set as an environment variable:

```python
from NiW.scraper import get_wikipedia_random_pages # Other scrapers function in roughly the same way

get_wikipedia_random_pages(
	experiment_id="name_of_the_experiment",
	limit="number_of_pages_you_want_to_get"
)
```

Or, you may implement a scraper on your own and save the content as JSON in the format:

```json
[
    {
        "title": "Page Title",
        "url": "https://example.com/page",
        "content": "Full page content..."
    }
]
```

Note that irrlevant content (like advertisements and links) has a negative impact on the queries generated, so it is highly recommended that you implement some website-specific content filters.

### Step 2: Generate Queries

Use the main pipeline to generate queries from your web content:

```python
from main import generate_query

# Generate all difficulty levels
queryset_specs = generate_query(
    experiment_id="name_of_the_experiment",
    web_content_path="path/to/your/content.json",
    mode="all",  # "easy", "medium", "hard", or "all"
    top_k=3,     # Number of claims per query
    url_list=None # Optional: limit to specific URLs
)
```

This will create:
- `experiments/{experiment_id}/querysets/queryset_easy_{timestamp}.json`
- `experiments/{experiment_id}/querysets/queryset_medium_{timestamp}.json`
- `experiments/{experiment_id}/querysets/queryset_hard_{timestamp}.jsonn`

#### Example Query Structure

```json
{
    "id": 0,
    "context": {
        "title": "Article Title",
        "url": "https://source.com/article",
        "content": "Full article content..."
    },
    "raw_questions": [
        "Someone discovered something in a certain year.",
        "A certain method achieved specific results."
    ],
    "ground_truth": [
        "Marie Curie discovered radium in 1898.",
        "The new method achieved 95% accuracy."
    ]
}
```