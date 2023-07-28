import re
import json

from spacy.lang.en import English

from lm_kbc.common.utils import relation_to_qa_prompt, get_contexts_from_file
from lm_kbc.common.file_io import read_lm_kbc_jsonl

nlp = English()
tokenizer = nlp.tokenizer


def len_tokens(i, j, text, strings, indices):
    """This function calculates the length in tokens between word in pos i and
    word in pos j"""
    word_i = strings[indices.index(i)]
    doc = nlp(text[i + len(word_i) : j])
    token_length = len(doc)
    return token_length


def find_span_insensitive_with_missing(paragraph, strings):
    n = len(strings)
    paragraph = paragraph.lower()
    strings = [s.lower() for s in strings]
    indices = [float("inf")] * n
    for i in range(n):
        match = re.search(strings[i], paragraph)
        if match:
            indices[i] = match.start()

    non_inf_indices = [i for i in indices if i != float("inf")]
    if not non_inf_indices:
        return None

    sorted_indices = sorted(non_inf_indices)
    for i, index in enumerate(sorted_indices[:-1]):
        if (len_tokens(index, sorted_indices[i + 1], paragraph, strings, indices)) > 3:
            return None

    start_index = min(non_inf_indices)
    end_index = max(non_inf_indices) + len(strings[indices.index(max(non_inf_indices))])
    return (start_index, end_index, len(non_inf_indices))


def main():
    for percentage in [5, 10, 20]:
        for k in range(10):
            print(f"{str(k)}/10")
            sample = read_lm_kbc_jsonl(
                f"./data/processed/train/train2-{str(percentage)}-{str(k)}.jsonl"
            )
            contexts = get_contexts_from_file(
                "./data/processed/train/contexts/contexts_train.json"
            )
            my_list = []

            for item in sample:
                context = contexts[item["Relation"]][item["SubjectEntity"]]["contexts"][
                    0
                ]
                question = relation_to_qa_prompt[item["Relation"]].replace(
                    "{}", item["SubjectEntity"]
                )
                if item["ObjectEntities"] == []:
                    training_example = {
                        "context": context,
                        "question": question,
                        "answers": {"text": [], "answer_start": []},
                    }
                    my_list.append(
                        {"id": str(id(training_example)), **training_example}
                    )

                objects = [
                    label
                    for alternatives in item["ObjectEntities"]
                    for label in alternatives
                ]
                candidates = []
                for context in contexts[item["Relation"]][item["SubjectEntity"]][
                    "contexts"
                ][:10]:
                    spans = find_span_insensitive_with_missing(context, objects)
                    if spans:
                        candidates.append((context, spans))
                if len(candidates) > 0:
                    candidates_sorted = sorted(candidates, key=lambda a: a[1][2])
                    context = candidates_sorted[0][0]
                    spans = candidates_sorted[0][1]
                    training_example = {
                        "context": context,
                        "question": question,
                        "answers": {
                            "text": [context[spans[0] : spans[1]]],
                            "answer_start": [spans[0]],
                        },
                    }
                    my_list.append(
                        {"id": str(id(training_example)), **training_example}
                    )

            with open(
                "".join(
                    [
                        "data/processed/train/lm_kbc_train2_squad_",
                        f"{str(percentage)}-{str(k)}.json",
                    ]
                ),
                "a",
            ) as f:
                for result in my_list:
                    f.write(json.dumps(result) + "\n")

    sample = read_lm_kbc_jsonl("./data/processed/train/train2.jsonl")
    contexts = get_contexts_from_file(
        "./data/processed/train/contexts/contexts_train.json"
    )
    my_list = []
    for item in sample:
        context = contexts[item["Relation"]][item["SubjectEntity"]]["contexts"][0]
        question = relation_to_qa_prompt[item["Relation"]].replace(
            "{}", item["SubjectEntity"]
        )
        if item["ObjectEntities"] == []:
            training_example = {
                "context": context,
                "question": question,
                "answers": {"text": [], "answer_start": []},
            }
            my_list.append({"id": str(id(training_example)), **training_example})

        objects = [
            label for alternatives in item["ObjectEntities"] for label in alternatives
        ]
        candidates = []
        for context in contexts[item["Relation"]][item["SubjectEntity"]]["contexts"][
            :10
        ]:
            spans = find_span_insensitive_with_missing(context, objects)
            if spans:
                candidates.append((context, spans))
        if len(candidates) > 0:
            candidates_sorted = sorted(candidates, key=lambda a: a[1][2])
            context = candidates_sorted[0][0]
            spans = candidates_sorted[0][1]
            training_example = {
                "context": context,
                "question": question,
                "answers": {
                    "text": [context[spans[0] : spans[1]]],
                    "answer_start": [spans[0]],
                },
            }
            my_list.append({"id": str(id(training_example)), **training_example})

    with open("data/processed/train/lm_kbc_train2_squad.json", "a") as f:
        for result in my_list:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
