from NiW.scraper import *
from NiW.claim_extraction import *
from NiW.claim_dissimilarity import *
from NiW.llm_judge import *
from NiW.logger import get_logging_path, log_message
import time
import litellm

def generate_query(experiment_id, web_content_path: str, mode: Literal["easy", "medium", "hard", "all"] = "all", top_k: int = 3, url_list: list[str] = None):
    log_path = get_logging_path(experiment_id=experiment_id, directory="queryset_logs", prefix=f"queryset_{mode}")
    news_count = 1000
    # news_list = get_cnn_news(experiment_id=experiment_id, topic = "all", limit=news_count, url_list=url_list)
    page_list = []
    with open(web_content_path, "r") as f:
        raw_page_list = json.loads(f.read())
        for page in raw_page_list:
            page_list.append(QueryContextPage(**page))
    hard_query_list = []
    medium_query_list = []
    easy_query_list = []
    for i, page in enumerate(page_list):
        log_message("Page content extracted", f"Title: {page.title}\nURL: {page.url}\nContent: {page.content[:255]}", log_path)
        claims = extract_claims(page.content)
        log_message("Extracted claims", "\n".join(claims), log_path)
        question_candidates = sort_sentence_by_dissimilarity(claims, page.content)
        if mode == "hard" or mode == "all":
            claims = question_candidates[-1 * top_k:]
            log_message("Claims for hard query", "\n".join(claims), log_path)
            subquestion_list = formulate_questions(page.content, claims)
            log_message("Formulated hard query", "\n".join(subquestion_list), log_path)
            query = Query(context=page, ground_truth=claims, id = i, raw_questions=subquestion_list)
            if validate_query(query, log_path):
                hard_query_list.append(query)
        if mode == "medium" or mode == "all":
            middle_index = (int)(len(question_candidates) / 2)
            if top_k % 2 == 1:
                claims = question_candidates[middle_index - (int)(top_k / 2): middle_index + 1 + (int)(top_k / 2)]
            else:
                claims = question_candidates[middle_index - (int)(top_k / 2): middle_index + (int)(top_k / 2)]
            log_message("Claims for medium query", "\n".join(claims), log_path)
            subquestion_list = formulate_questions(page.content, claims)
            log_message("Formulated medium query", "\n".join(subquestion_list), log_path)
            query = Query(context=page, ground_truth=claims, id = i, raw_questions=subquestion_list)
            if validate_query(query, log_path):
                medium_query_list.append(query)
        if mode == "easy" or mode == "all":
            claims = question_candidates[:top_k]
            log_message("Claims for easy query", "\n".join(claims), log_path)
            subquestion_list = formulate_questions(page.content, claims)
            log_message("Formulated easy query", "\n".join(subquestion_list), log_path)
            query = Query(context=page, ground_truth=claims, id = i, raw_questions=subquestion_list)
            if validate_query(query, log_path):
                easy_query_list.append(query)
    queryset_specs = []
    if mode == "hard" or mode == "all":
        queryset_name = f"hard_{time.strftime('%Y%m%d_%H%M%S')}"
        savefile_hard_name = f"experiments/{str(experiment_id)}/querysets/queryset_{queryset_name}.json"
        queryset_specs.append({"name": queryset_name, "filename": savefile_hard_name})
        with open(savefile_hard_name, "w") as f:
            f.write(json.dumps([query.json() for query in hard_query_list], indent=4))
    if mode == "medium" or mode == "all":
        queryset_name = f"medium_{time.strftime('%Y%m%d_%H%M%S')}"
        savefile_medium_name = f"experiments/{str(experiment_id)}/querysets/queryset_{queryset_name}.json"
        queryset_specs.append({"name": queryset_name, "filename": savefile_medium_name})
        with open(savefile_medium_name, "w") as f:
            f.write(json.dumps([query.json() for query in medium_query_list], indent=4))
    if mode == "easy" or mode == "all":
        queryset_name = f"easy_{time.strftime('%Y%m%d_%H%M%S')}"
        savefile_easy_name = f"experiments/{str(experiment_id)}/querysets/queryset_{queryset_name}.json"
        queryset_specs.append({"name": queryset_name, "filename": savefile_easy_name})
        with open(savefile_easy_name, "w") as f:
            f.write(json.dumps([query.json() for query in easy_query_list], indent=4))
    return queryset_specs

def get_model_answer(experiment_id, queryset_path: str, queryset_name: str, model: Literal["oai", "gemini", "perplexity"]):
    # log_path = get_logging_path(directory="test_logs", prefix=f"test_{queryset_name}")
    with open(queryset_path, "r") as f:
        query_list = json.loads(f.read())
    test_result_list = []
    answer_list = []
    for i, query in enumerate(query_list):
        query = Query(id=query["id"], context=QueryContextPage(**query["context"]), ground_truth=query["ground_truth"], raw_questions=query["raw_questions"])
        query.question = QUERY_TEMPLATE.format(question="\n\n".join(query.raw_questions))
        print("\n\n\n-----------------------------------------------------------------------------------------------------------\n")
        print("\n\n".join(query.raw_questions) + "\n\n")
        print("\n\n".join(query.ground_truth) + "\n\n")
        # log_message("Query Info", f"{query.context.url}\n{query.context.title}\n{query.question}\n{query.ground_truth}", log_path)
        search, source = get_web_search_answer(query.question, model=model)
        test_result_list.append({"id": query.id, "source": source, "search": search})
        # log_message("Web Search Result", answers + "\n\n" + source, log_path)
    savefile_name = f"experiments/{str(experiment_id)}/test_raw_responses/raw_responses_{queryset_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(savefile_name, "w") as f:
        f.write(json.dumps(test_result_list, indent=4))
    return savefile_name

def judge_model_answer(experiment_id, raw_results_path: str, queryset_path: str, queryset_name: str):
    log_path = get_logging_path(experiment_id=experiment_id, directory="test_logs", prefix=f"test_{queryset_name}")
    with open(queryset_path, "r") as f:
        query_list = json.loads(f.read())
    with open(raw_results_path, "r") as f:
        raw_results = json.loads(f.read())
    test_result_list = []
    for i, query in enumerate(query_list):
        query = Query(id=query["id"], context=QueryContextPage(**query["context"]), ground_truth=query["ground_truth"], raw_questions=query["raw_questions"])
        query.question = QUERY_TEMPLATE.format(question="\n\n".join(query.raw_questions))
        raw_questions_string = "\n\n".join(query.raw_questions)
        ground_truth_string = "\n\n".join(query.ground_truth)
        log_message("Query Info", f"{query.context.url} \n{query.context.title} \n{raw_questions_string} \n{ground_truth_string}", log_path)
        source = raw_results[i]["source"]
        search = ""
        if raw_results[i].get("search", None) is not None:
            search = json.dumps(raw_results[i]["search"], indent=4)
        log_message("Web Search Result", source, log_path)
        if "no source extracted" in source.lower() or "no source found" in source.lower():
            log_message("Source Check", "No source provided, skipping further checks.", log_path)
            test_result_list.append({"id": query.id, "correct": False, "reason": "No source provided"})
            continue
        page_content = remove_links_from_markdown(get_page_content(source))
        raw_questions = query.raw_questions
        source_accepted = True
        for i, item in enumerate(raw_questions):
            accept, reason = check_source(page_content, item)
            if accept:
                log_message("Source Check", f"The claim '{item}' is supported by the citation.\n" + reason, log_path)
            else:
                source_accepted = False
                log_message("Source Check", f"The claim '{item}' is NOT supported by the citation.\n" + reason, log_path)
        if not source_accepted:
            log_message("Judging Result", "The source is incorrect", log_path)
            test_result_list.append({"id": query.id, "correct": False, "reason": "The source is incorrect"})
            continue
        found_original_webpage = True
        for i, item in enumerate(query.ground_truth):
            accept, reason = check_source_exact(query.context.content, item)
            if accept:
                log_message("Exact Claim Check", f"The claim '{item}' is supported by the original webpage.\n" + reason, log_path)
            else:
                found_original_webpage = False
                log_message("Exact Claim Check", f"The claim '{item}' is NOT supported by the original webpage.\n" + reason, log_path)
        if found_original_webpage:
            test_result_list.append({"id": query.id, "correct": True, "reason": "Found original webpage"})
        else:
            test_result_list.append({"id": query.id, "correct": True, "reason": "Found another correct webpage"})

    def postprocess_test_result(test_result_list: dict):
        correct_count = 0
        ground_truth_match_count = 0
        criteria_match_count = 0
        invalid_source_count = 0
        wrong_webpage_count = 0
        for i, result in enumerate(test_result_list):
            if result["correct"]:
                correct_count += 1
            if result["reason"] == "Found original webpage":
                ground_truth_match_count += 1
            elif result["reason"] == "Found another correct webpage":
                criteria_match_count += 1
            elif result["reason"] == "No source provided":
                invalid_source_count += 1
            elif result["reason"] == "The source is incorrect":
                wrong_webpage_count += 1
        new_result = {
            "item_count": len(test_result_list),
            "accuracy": correct_count / len(test_result_list),
            "correct_count": correct_count,
            "ground_truth_match_count": ground_truth_match_count,
            "criteria_match_count": criteria_match_count,
            "invalid_source_count": invalid_source_count,
            "wrong_webpage_count": wrong_webpage_count,
            "detail": test_result_list
        }
        return new_result
    new_result = postprocess_test_result(test_result_list)
    savefile_name = f"experiments/{str(experiment_id)}/test_results/results_{queryset_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(savefile_name, "w") as f:
        f.write(json.dumps(new_result, indent=4))
    correct_count = 0
    for result in test_result_list:
        if result["correct"]:
            correct_count = correct_count + 1
    print(f"Acc: {correct_count/len(test_result_list)}")
    return savefile_name

def validate_query(query: Query, log_path):
    valid = True
    for i, item in enumerate(query.raw_questions):
        accepted, reason = check_source(query.context.content, item)
        if not accepted:
            valid = False
            log_message("Validation Result", f"subquestion '{item}' is NOT supported by the context.\n" + reason, log_path)
    return valid


def full_pipeline(experiment_id, web_content_path, mode, model: Literal["oai", "gemini", "perplexity"], queryset_specs = None):
    #create directory if not exists
    os.makedirs(f"experiments/{experiment_id}/querysets", exist_ok=True)
    os.makedirs(f"experiments/{experiment_id}/test_raw_responses", exist_ok=True)
    os.makedirs(f"experiments/{experiment_id}/test_results", exist_ok=True)
    os.makedirs(f"experiments/{experiment_id}/web_contents", exist_ok=True)
    os.makedirs(f"experiments/{experiment_id}/logs", exist_ok=True)
    os.makedirs(f"experiments/{experiment_id}/logs/queryset_logs", exist_ok=True)
    os.makedirs(f"experiments/{experiment_id}/logs/test_logs", exist_ok=True)
    top_k = 3
    if queryset_specs is None:
        queryset_specs = generate_query(experiment_id, web_content_path, mode, top_k)
    for spec in queryset_specs:
        print(f"Generated queryset: {spec['name']}")
        raw_results_path = get_model_answer(experiment_id, spec["filename"], spec["name"], model)
        print(f"Raw results saved to: {raw_results_path}")
        test_result_path = judge_model_answer(experiment_id, raw_results_path, spec["filename"], spec["name"])
        print(f"Test results saved to: {test_result_path}")
    return queryset_specs


if __name__ == "__main__":
    # querysets = os.listdir("experiments/deepresearcher/test_raw_responses")
    querysets = ["lonelyplanet_easy.json", "lonelyplanet_hard.json", "lonelyplanet_medium.json"]
    for queryset_name in querysets:
        judge_model_answer("deepresearcher", "experiments/deepresearcher/test_raw_responses/" + queryset_name, "querysets/" + queryset_name, queryset_name[:-5])