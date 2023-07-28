import re
import argparse

from tqdm import tqdm
from numpy import mean
from transformers import pipeline

from lm_kbc.common.utils import (
    read_lm_kbc_jsonl,
    relation_to_qa_prompt,
    get_contexts_from_file,
)
from lm_kbc.common.evaluate import evaluate_per_sr_pair, combine_scores_per_relation
from lm_kbc.modeling.zero_shot_entailment import save_results, relation_to_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot Kowledge Base Construction with Question Answering."
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        type=str,
        default="bert-large-cased-whole-word-masking-finetuned-squad",
        help='Model to use question answering \
                (default "bert-large-cased-whole-word-masking-finetuned-squad")',
    )
    parser.add_argument(
        "--k_sentences",
        metavar="k",
        type=int,
        default=3,
        help="Number of contexts to use for question answering (3 by default)",
    )
    parser.add_argument(
        "--qa_threshold",
        type=float,
        default=0.5,
        help="Score threshold in question answser (default 0.5)",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/raw/lm-kbc/dataset/data/dev.jsonl",
        help='Path to the input (default "./data/raw/lm-kbc/dataset/data/dev.jsonl")',
    )
    parser.add_argument(
        "--input_path_dev_2",
        type=str,
        default="./data/raw/lm-kbc/dataset/data/dev.jsonl",
        help='Path to the development 2 dataset (default "./data/raw/lm-kbc/dataset/\
                data/dev.jsonl")',
    )
    parser.add_argument(
        "--contexts_path",
        type=str,
        default="./data/processed/dev/contexts/contexts.json",
        help='Path to the contexts (default "./data/processed/dev/contexts\
        /contexts.json")',
    )
    parser.add_argument(
        "--contexts_train_path",
        type=str,
        default="./data/processed/train/contexts/contexts_train.json",
        help='Path to the train contexts (default "./data/processed/train/contexts\
        /contexts_train.json")',
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/raw/lm-kbc/dataset/data/output.jsonl",
        help='Path to the output (default "./data/raw/lm-kbc/dataset/data\
        /output.jsonl")',
    )
    parser.add_argument(
        "--calculate_qa_threshold",
        default=False,
        action="store_true",
        help="Whether to calculate question answering threshold (This parameter regulates the \
                ammount of candidates per relation) (default False if not present)",
    )

    args = parser.parse_args()

    return args


def question_answering(args, qa_model, sents, row):
    subjectEntity = row["SubjectEntity"]
    relation = row["Relation"]
    question = relation_to_qa_prompt[relation].replace("{}", subjectEntity)
    answers = {}
    if len(sents) > 0:
        res = qa_model(question=[question] * len(sents), context=sents)
        if type(res) == dict:
            res = [res]
        for res_item in res:
            internal_answers = [
                res.strip()
                for res in re.split(r"\,|and", res_item["answer"])
                if res.strip()
            ]
            for internal_answer in internal_answers:
                if internal_answer in answers:
                    answers[internal_answer] += [res_item["score"]]
                else:
                    answers[internal_answer] = [res_item["score"]]
    answers = {label: mean(score) for label, score in answers.items()}

    return answers


def calculate_qa_threshold(args, qa_model, contexts, contexts_train):
    print("**Calculating qa threshold**")
    input_rows = read_lm_kbc_jsonl(args.input_path)
    input_rows_dev_2 = read_lm_kbc_jsonl(args.input_path_dev_2)
    predict_results = []
    for row in tqdm(input_rows_dev_2):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        sents = contexts_train[relation][subjectEntity]["contexts"][: args.k_sentences]
        answers = question_answering(args, qa_model, sents, row)
        predict_results.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "PredictAnswers": answers,
                "ObjectEntities": row["ObjectEntities"],
            }
        )

    best_qa_threshold = {rel: None for rel in relation_to_prompt}
    best_qa_threshold_scores = {rel: None for rel in relation_to_prompt}

    begin = 0.01
    end = 0.99
    step = 0.01

    curr_t_threshold = begin
    tidx_threshold = 0

    while curr_t_threshold < end:
        if tidx_threshold % 10 == 0:
            print(f"tidx_threshold: {tidx_threshold};")
        for rel in relation_to_prompt:
            curr_allentry_results = []
            for res in predict_results:
                if res["Relation"] != rel:
                    continue

                if len(res["PredictAnswers"]) == 0:
                    res["PredictAnswers"] = {}
                curr_objs = set(
                    [
                        label
                        for label, score in res["PredictAnswers"].items()
                        if score > curr_t_threshold
                    ]
                )
                curr_res = {
                    "SubjectEntity": res["SubjectEntity"],
                    "ObjectEntities": curr_objs,
                    "Relation": res["Relation"],
                }
                curr_allentry_results.append(curr_res)
            # Evaluate
            scores_per_sr_pair = evaluate_per_sr_pair(
                curr_allentry_results, input_rows_dev_2
            )
            scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)
            if (
                best_qa_threshold_scores[rel] is None
                or scores_per_relation[rel]["f1"] > best_qa_threshold_scores[rel]["f1"]
            ):
                best_qa_threshold_scores[rel] = scores_per_relation[rel]
                best_qa_threshold[rel] = curr_t_threshold
        curr_t_threshold += step
        tidx_threshold += 1

    print("**Best qa threshold**:", best_qa_threshold)
    best_candidates = []
    for row in tqdm(input_rows):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        sents = contexts[relation][subjectEntity]["contexts"][: args.k_sentences]
        answers = question_answering(args, qa_model, sents, row)
        if len(answers) == 0:
            answers = {}
        curr_objs = set(
            [
                label
                for label, score in answers.items()
                if score > best_qa_threshold[relation]
            ]
        )
        best_candidates.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "ObjectEntities": list(curr_objs),
            }
        )
    print("**Using candidates with the best qa threshold**")
    return best_candidates


def main(args):
    qa_model = pipeline("question-answering", model=args.model, device=0)
    contexts = get_contexts_from_file(args.contexts_path)
    if args.calculate_qa_threshold:
        contexts_train = get_contexts_from_file(args.contexts_train_path)
        predict_results = calculate_qa_threshold(
            args, qa_model, contexts, contexts_train
        )

    else:
        input_rows = read_lm_kbc_jsonl(args.input_path)
        predict_results = []
        for row in tqdm(input_rows):
            subjectEntity = row["SubjectEntity"]
            relation = row["Relation"]
            sents = contexts[relation][subjectEntity]["contexts"][: args.k_sentences]
            answers = question_answering(args, qa_model, sents, row)
            filtered_answers = set(
                [label for label, score in answers.items() if score > args.qa_threshold]
            )
            predict_results.append(
                {
                    "SubjectEntity": subjectEntity,
                    "Relation": relation,
                    "ObjectEntities": list(filtered_answers),
                }
            )
    save_results(args, predict_results)


if __name__ == "__main__":
    main(parse_args())
