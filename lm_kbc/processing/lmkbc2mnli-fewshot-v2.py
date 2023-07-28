import re
import json
import random

from transformers import pipeline

from lm_kbc.common.utils import (
    get_element_symbol,
    get_candidates_from_lm,
    get_contexts_from_file,
    apply_fixed_candidates_filter,
)
from lm_kbc.common.file_io import read_lm_kbc_jsonl
from lm_kbc.modeling.zero_shot_entailment import relation_to_prompt

unmasker = pipeline("fill-mask", model="bert-large-cased", device=0)


def lmkbc2mnli(input_dataset, output_dataset):

    sample = read_lm_kbc_jsonl(input_dataset)
    contexts = get_contexts_from_file(
        "./data/processed/train/contexts/contexts_train.json"
    )
    random.seed(42)
    my_list = []
    objects = {
        sample_ii[0]
        for sample_i in sample
        for sample_ii in sample_i["ObjectEntities"]
        if len(sample_ii) > 0
    }
    for item in sample:
        context = contexts[item["Relation"]][item["SubjectEntity"]]["contexts"][0]
        candidates = get_candidates_from_lm(
            item["SubjectEntity"], item["Relation"], lm_candidates=unmasker
        )
        candidates_filtered = apply_fixed_candidates_filter(
            candidates,
            item["SubjectEntity"],
            item["Relation"],
            "./data/processed/train/contexts/contexts_train.json",
        )
        candidates_filtered = {object.lower() for object in candidates_filtered}
        candidates = (
            list(candidates_filtered)
            + list(
                {candidate.lower() for candidate in candidates} - candidates_filtered
            )
            + list(random.sample(objects, len(objects)))
        )
        object_list = [
            subitem
            for object in item["ObjectEntities"]
            if len(object) > 0
            for subitem in object
        ]
        wrong_candidates = [
            candidate for candidate in candidates if candidate not in object_list
        ]
        wrong_candidate_index = 0
        if item["ObjectEntities"] == []:
            my_list.append(
                {
                    "premise": context,
                    "hypothesis": "".join(
                        [
                            f"{item['SubjectEntity']} ",
                            relation_to_prompt[item["Relation"]],
                        ]
                    ).format(wrong_candidates[0]),
                    "label": "contradiction",
                }
            )
        for object in item["ObjectEntities"]:
            if len(object) == 0:
                continue
            i = 1
            while (
                (
                    item["Relation"] == "ChemicalCompoundElement"
                    and not re.search(get_element_symbol(object[0]), context)
                )
                or not re.search("|".join(object), context.lower())
            ) and i < 10:
                context = contexts[item["Relation"]][item["SubjectEntity"]]["contexts"][
                    i
                ]
                i += 1
            if i < 10:
                my_list.append(
                    {
                        "premise": context,
                        "hypothesis": "".join(
                            [
                                f"{item['SubjectEntity']} ",
                                relation_to_prompt[item["Relation"]],
                            ]
                        ).format(object[0]),
                        "label": "entailment",
                    }
                )
                my_list.append(
                    {
                        "premise": context,
                        "hypothesis": "".join(
                            [
                                f"{item['SubjectEntity']} ",
                                relation_to_prompt[
                                    random.sample(
                                        set(relation_to_prompt.keys())
                                        - {item["Relation"]},
                                        1,
                                    )[0]
                                ],
                            ]
                        ).format(object[0]),
                        "label": "neutral",
                    }
                )
                my_list.append(
                    {
                        "premise": context,
                        "hypothesis": "".join(
                            [
                                f"{item['SubjectEntity']} ",
                                relation_to_prompt[item["Relation"]],
                            ]
                        ).format(wrong_candidates[wrong_candidate_index]),
                        "label": "contradiction",
                    }
                )
                wrong_candidate_index += 1
    with open(output_dataset, "a") as f:
        for result in my_list:
            f.write(json.dumps(result) + "\n")


for dataset in ["train2", "dev2"]:
    for percentage in [5, 10, 20]:
        for k in range(10):
            input_dataset = (
                f"./data/processed/train/{dataset}-{str(percentage)}-{str(k)}.jsonl"
            )
            output_dataset = "".join(
                [
                    f"data/processed/train/lm_kbc_{dataset}_mnli_",
                    f"{str(percentage)}-{str(k)}-v2.json",
                ]
            )
            lmkbc2mnli(input_dataset, output_dataset)
    input_dataset = f"./data/processed/train/{dataset}.jsonl"
    output_dataset = f"data/processed/train/lm_kbc_{dataset}_mnli-v2.json"
    lmkbc2mnli(input_dataset, output_dataset)
