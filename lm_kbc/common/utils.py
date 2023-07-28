import json
import string

import spacy

from tqdm import tqdm
from duckduckgo_search import ddg

from lm_kbc.common.file_io import read_lm_kbc_jsonl

nlp = spacy.load("en_core_web_trf")
unmasker = None

relation_to_prompt = {
    "CountryBordersWithCountry": "shares a land border with",
    "CountryOfficialLanguage": "official language",
    "StateSharesBorderState": "shares borders with",
    "RiverBasinsCountry": "country basins",
    "ChemicalCompoundElement": "elements",
    "PersonLanguage": "languages spoken",
    "PersonProfession": "profession",
    "PersonInstrument": "plays instrument",
    "PersonEmployer": "employer",
    "PersonPlaceOfDeath": "place of death",
    "PersonCauseOfDeath": "cause of death",
    "CompanyParentOrganization": "'s parent organization",
}

relation_to_qa_prompt = {
    "CountryBordersWithCountry": "What country borders with {}?",
    "CountryOfficialLanguage": "What official language of {}?",
    "StateSharesBorderState": "What state shares borders with {}?",
    "RiverBasinsCountry": "What countries in basin of {}?",
    "ChemicalCompoundElement": "What chemical elements of {}?",
    "PersonLanguage": "What languages speak {}?",
    "PersonProfession": "What is the profession of {}?",
    "PersonInstrument": "What instruments plays {}?",
    "PersonEmployer": "What company employs {}?",
    "PersonPlaceOfDeath": "Where did {} die?",
    "PersonCauseOfDeath": "What is the cause of death of {}?",
    "CompanyParentOrganization": "What is the parent organization of {}?",
}

relation_filter = {
    "CompanyParentOrganization": ["ORG"],
    "PersonEmployer": ["ORG"],
    "PersonPlaceOfDeath": ["GPE"],
    "StateSharesBorderState": ["GPE"],
    "CountryBordersWithCountry": ["GPE"],
    "RiverBasinsCountry": ["GPE"],
}

banned_words = [
    "I",
    "you",
    "he",
    "she",
    "they",
    "it",
    "we",
    "me",
    "him",
    "her",
    "us",
    "them",
    "mine",
    "yours",
    "his",
    "hers",
    "ours",
    "theirs",
    "myself",
    "yourself",
    "himself",
    "herself",
    "itself",
    "ourselves",
    "yourselves",
    "themselves",
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "many",
    "some",
    "much",
    "most",
    "any",
    "home",
]


def create_prompt(subject_entity: str, relation: str, mask_token: str) -> str:
    """
    Depending on the relation, we fix the prompt
    """

    prompt = mask_token

    if relation == "CountryBordersWithCountry":
        prompt = f"{subject_entity} shares border with {mask_token}."
    elif relation == "CountryOfficialLanguage":
        prompt = f"The official language of {subject_entity} is {mask_token}."
    elif relation == "StateSharesBorderState":
        prompt = f"{subject_entity} shares border with {mask_token} state."
    elif relation == "RiverBasinsCountry":
        prompt = f"{subject_entity} river basins in {mask_token}."
    elif relation == "ChemicalCompoundElement":
        prompt = f"{subject_entity} consists of {mask_token}, " f"which is an element."
    elif relation == "PersonLanguage":
        prompt = f"{subject_entity} speaks in {mask_token}."
    elif relation == "PersonProfession":
        prompt = f"{subject_entity} is a {mask_token} by profession."
    elif relation == "PersonInstrument":
        prompt = f"{subject_entity} plays {mask_token}, which is an instrument."
    elif relation == "PersonEmployer":
        prompt = (
            f"{subject_entity} is an employer at {mask_token}, " f"which is a company."
        )
    elif relation == "PersonPlaceOfDeath":
        prompt = f"{subject_entity} died at {mask_token}."
    elif relation == "PersonCauseOfDeath":
        prompt = f"{subject_entity} died due to {mask_token}."
    elif relation == "CompanyParentOrganization":
        prompt = f"The parent organization of {subject_entity} is {mask_token}."

    return prompt


def get_candidates(subject, relation):
    search_text = subject + " " + relation_to_prompt[relation]
    results = ddg(search_text)
    doc = nlp("\n".join([result["title"] + " " + result["body"] for result in results]))
    candidates = []
    if relation in relation_filter.keys():
        for entity in doc.ents:
            if entity.label_ in relation_filter[relation]:
                candidates.append(entity.text)
    else:
        for entity in doc.ents:
            candidates.append(entity.text)

    return list(dict.fromkeys(candidates))


def get_contexts_from_file(contexts_file):
    with open(contexts_file) as f:
        contexts_json = json.load(f)
    return contexts_json


def get_candidates_from_contexts(subject, relation, contexts_file):
    results = get_contexts_from_file(contexts_file)[relation][subject]["contexts"]
    doc = nlp("\n".join(results[:10]))
    candidates = []
    if relation in relation_filter.keys():
        for entity in doc.ents:
            if entity.label_ in relation_filter[relation]:
                candidates.append(entity.text)
    else:
        for entity in doc.ents:
            candidates.append(entity.text)

    return list(dict.fromkeys(candidates))


def get_candidates_from_lm(
    subject,
    relation,
    top_p=0.8,
    return_scores=False,
    context=None,
    lm_candidates=None,
    filter_before=False,
):
    unmasker = lm_candidates
    prompt = create_prompt(subject, relation, "[MASK]")
    if context:
        prompt = context + "[SEP]" + prompt
    cumulative_sum = 0
    candidates = []
    for item in unmasker(prompt, top_k=100):
        if filter_before and (
            item["token_str"]
            in banned_words
            # or item["token_str"] in string.punctuation
        ):
            continue
        candidates.append(
            item["token_str"]
            if not return_scores
            else (item["token_str"], item["score"])
        )
        cumulative_sum += item["score"]
        if cumulative_sum > top_p:
            break
    return candidates


def get_candidates_from_lm_threshold(
    subject,
    relation,
    threshold=0.5,
    return_scores=False,
    lm_candidates=None,
    filter_before=False,
    filter_punctuation=False,
):
    unmasker = lm_candidates
    prompt = create_prompt(subject, relation, "[MASK]")
    candidates = []
    for item in unmasker(prompt, top_k=100):
        if item["score"] < threshold:
            break

        if filter_before and (
            item["token_str"] in banned_words
            or (filter_punctuation and item["token_str"] in string.punctuation)
        ):
            continue

        candidates.append(
            item["token_str"]
            if not return_scores
            else (item["token_str"], item["score"])
        )

    return candidates


def filter_candidates_with_top_p(
    candidates,
    top_p=0.8,
    return_scores=False,
):
    cumulative_sum = 0.0
    filtered_candidates = []

    for item_label, item_score in candidates:
        filtered_candidates.append(
            item_label if not return_scores else (item_label, item_score)
        )
        cumulative_sum += item_score
        if cumulative_sum > top_p:
            break
    return filtered_candidates


def filter_candidates_with_epsilon(
    candidates,
    epsilon=0.01,
    return_scores=False,
):
    last_probability = 0.0
    filtered_candidates = []

    for item_label, item_score in candidates:
        if abs(last_probability - item_score) >= epsilon:
            filtered_candidates.append(
                item_label if not return_scores else (item_label, item_score)
            )
            last_probability = item_score
        else:
            break
    return filtered_candidates


def filter_candidates_with_lm_thres(
    candidates,
    threshold=0.01,
    return_scores=False,
):
    filtered_candidates = []

    for item_label, item_score in candidates:
        if item_score >= threshold:
            filtered_candidates.append(
                item_label if not return_scores else (item_label, item_score)
            )
        else:
            break
    return filtered_candidates


def get_countries_list():
    with open("./data/processed/dev/candidates/countries.json", "r") as file:
        res_json = json.load(file)
    return (
        [elem["name"] for elem in res_json["geonames"]]
        if "geonames" in res_json
        else None
    )


def get_languages_list():
    with open("./data/processed/dev/candidates/languages_query.json", "r") as file:
        data = json.load(file)
    return [item["itemLabel"] for item in data]


def get_elements_list():
    with open("./data/processed/dev/candidates/elements_query.json", "r") as file:
        data = json.load(file)
    return [item["elementLabel"] for item in data]


def get_elements_with_symbol_list():
    with open("./data/processed/dev/candidates/elements_query.json", "r") as file:
        data = json.load(file)
    return [(item["elementLabel"], item["symbolLabel"]) for item in data]


def get_causes_of_death():
    input_rows = read_lm_kbc_jsonl("./data/raw/lm-kbc/dataset/data/train.jsonl")
    causes = set(
        [
            item
            for input_row in input_rows
            if input_row["Relation"] == "PersonCauseOfDeath"
            for items in input_row["ObjectEntities"]
            for item in items
        ]
    )
    return list(causes)


def get_causes_of_death_list():
    with open("./data/processed/dev/candidates/causes_of_death.json", "r") as file:
        data = json.load(file)
    return [item["causeOfDeathLabel"] for item in data]


def get_instruments():
    input_rows = read_lm_kbc_jsonl("./data/raw/lm-kbc/dataset/data/train.jsonl")
    causes = set(
        [
            item
            for input_row in input_rows
            if input_row["Relation"] == "PersonInstrument"
            for items in input_row["ObjectEntities"]
            for item in items
        ]
    )
    return list(causes)


def get_instruments_list():
    with open("./data/processed/dev/candidates/instruments.json", "r") as file:
        data = json.load(file)
    return [item["instrumentLabel"].lower() for item in data]


def get_professions():
    input_rows = read_lm_kbc_jsonl("./data/raw/lm-kbc/dataset/data/train.jsonl")
    causes = set(
        [
            item
            for input_row in input_rows
            if input_row["Relation"] == "PersonProfession"
            for items in input_row["ObjectEntities"]
            for item in items
        ]
    )
    return list(causes)


def get_professions_list():
    with open("./data/processed/dev/candidates/professions.json", "r") as file:
        data = json.load(file)
    return [item["professionLabel"] for item in data]


def get_fixed_candidates(expand_candidates=False, elements_with_symbol=False):
    return {
        "countries": get_countries_list(),
        "languages": get_languages_list(),
        "elements": get_elements_with_symbol_list()
        if elements_with_symbol
        else get_elements_list(),
        "causes_of_death": list(set(get_causes_of_death() + get_causes_of_death_list()))
        if expand_candidates
        else get_causes_of_death(),
        "instruments": list(set(get_instruments() + get_instruments_list()))
        if expand_candidates
        else get_instruments(),
        "professions": list(set(get_professions() + get_professions_list()))
        if expand_candidates
        else get_professions(),
    }


def get_contexts(input_path, output_path):
    input_rows = read_lm_kbc_jsonl(input_path)
    contexts_dict = {key: {} for key in relation_to_prompt.keys()}
    for input_row in tqdm(input_rows):
        subject = input_row["SubjectEntity"]
        relation = input_row["Relation"]
        if relation not in relation_to_prompt.keys():
            continue
        search_text = subject + " " + relation_to_prompt[relation]
        results = ddg(search_text)
        contexts = [result["title"] + " " + result["body"] for result in results]
        contexts_dict[relation][subject] = {
            "search_text": search_text,
            "contexts": contexts,
        }

    with open(output_path, "w") as outfile:
        json.dump(contexts_dict, outfile)


def get_contexts_different_searchs(input_path, output_path):
    input_rows = read_lm_kbc_jsonl(input_path)
    contexts_dict = {key: {} for key in relation_to_prompt.keys()}
    for input_row in tqdm(input_rows):
        subject = input_row["SubjectEntity"]
        relation = input_row["Relation"]
        if relation not in relation_to_prompt.keys():
            continue
        contexts_dict[relation][subject] = {}
        for prompt in relation_to_prompt[relation]:
            search_text = subject + " " + prompt
            results = ddg(search_text)
            contexts = [result["title"] + " " + result["body"] for result in results]
            contexts_dict[relation][subject][prompt] = contexts

    with open(output_path, "w") as outfile:
        json.dump(contexts_dict, outfile)


def apply_fixed_candidates_filter(
    candidates_labels,
    subject,
    relation,
    contexts_path,
    return_scores=False,
    context_sents=None,
):
    final_candidates = []
    if relation in relation_filter:
        candidates_from_contexts = get_candidates_from_contexts(
            subject, relation, contexts_path
        )
        candidates_from_contexts_lower = [
            candidate.lower() for candidate in candidates_from_contexts
        ]
        for candidate in candidates_labels:
            score = candidate[1] if return_scores else None
            candidate = candidate[0] if return_scores else candidate
            if (not candidate.lower() == subject.lower()) and (
                candidate.lower() in candidates_from_contexts_lower
            ):
                final_candidates.append(
                    (candidate, score) if return_scores else candidate
                )
    else:
        if context_sents:
            contexts = context_sents
        else:
            contexts = get_contexts_from_file(contexts_path)[relation][subject][
                "contexts"
            ]
        context_text = "\n".join(contexts[:10]).lower()
        for candidate in candidates_labels:
            score = candidate[1] if return_scores else None
            candidate = candidate[0] if return_scores else candidate
            if candidate.lower() in context_text:
                final_candidates.append(
                    (candidate, score) if return_scores else candidate
                )
    return final_candidates


elems_with_symbols = get_elements_with_symbol_list()


def get_element_symbol(elem):
    for elem_pair in elems_with_symbols:
        if elem == "caesium":
            elem = "cesium"
        if elem.lower() == elem_pair[0].lower():
            return elem_pair[1]
    raise Exception(elem, "not found")


contexts_rso = {}


def load_contexts_rso(input_path):
    global contexts_rso
    with open(input_path) as f:
        contexts_rso = json.load(f)


def save_contexts_rso(output_path):
    with open(output_path, "w") as outfile:
        json.dump(contexts_rso, outfile)


def get_contexts_sro(subject, relation, objectEntity):
    if (
        (relation in contexts_rso)
        and (subject in contexts_rso[relation])
        and (objectEntity in contexts_rso[relation][subject])
    ):
        return contexts_rso[relation][subject][objectEntity]
    search_text = f"{subject} {relation_to_prompt[relation]} {objectEntity}"
    results = ddg(search_text)
    contexts = [result["title"] + " " + result["body"] for result in results]
    if relation in contexts_rso:
        if subject in contexts_rso[relation]:
            contexts_rso[relation][subject][objectEntity] = contexts[:10]
        else:
            contexts_rso[relation][subject] = {objectEntity: contexts[:10]}
    else:
        contexts_rso[relation] = {subject: {objectEntity: contexts[:10]}}
    return contexts
