import json
import argparse

from tqdm import tqdm
from numpy import mean
from transformers import pipeline

from lm_kbc.common.utils import (
    get_fixed_candidates,
    get_contexts_from_file,
    get_candidates_from_contexts,
    apply_fixed_candidates_filter,
    filter_candidates_with_lm_thres,
    get_candidates_from_lm_threshold,
)
from lm_kbc.common.file_io import read_lm_kbc_jsonl
from lm_kbc.common.evaluate import evaluate_per_sr_pair, combine_scores_per_relation

relation_to_prompt = {
    "CountryBordersWithCountry": "shares a land border with {}",
    "CountryOfficialLanguage": "official language is {}",
    "StateSharesBorderState": "shares borders with {}",
    "RiverBasinsCountry": "river basins {}",
    "ChemicalCompoundElement": "consists of {}",
    "PersonLanguage": "speaks {}",
    "PersonProfession": "is a {}",
    "PersonInstrument": "plays {}",
    "PersonEmployer": "'s employer is {}",
    "PersonPlaceOfDeath": "'s place of death is {}",
    "PersonCauseOfDeath": "'s cause of death is {}",
    "CompanyParentOrganization": "'s parent organization is {}",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot Kowledge Base Construction with entailment."
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        type=str,
        default="cross-encoder/nli-deberta-v3-xsmall",
        help='Model to use entailement (default "cross-encoder/nli-deberta-v3-xsmall")',
    )
    parser.add_argument(
        "--k_sentences",
        metavar="k",
        type=int,
        default=3,
        help="Number of contexts to use for entailment (3 by default)",
    )
    parser.add_argument(
        "--entailment_prob_threshold",
        metavar="p",
        type=float,
        default=0.5,
        help="Probability threshold in entailment to accept candidate \
        as answer (default 0.5)",
    )
    parser.add_argument(
        "--use_candidates_fixed",
        default=False,
        action="store_true",
        help="Wether to use the prefixed list of candidates for some \
        relations (default False if not present)",
    )
    parser.add_argument(
        "--filter_fixed_candidates",
        default=False,
        action="store_true",
        help="Wether to apply the heuristic filter for fixed candidates \
        and for lm candidates (default False if not present)",
    )
    parser.add_argument(
        "--candidates_generation",
        type=str,
        default="from_contexts",
        help="How to generate the candidates: extract them from context/prefixed \
        list (from_contexts) OR use a lm to generate them (from_lm) \
        (default from_contexts)",
    )
    parser.add_argument(
        "--expand_candidates",
        default=False,
        action="store_true",
        help="Expand fixed candidates for some relations (default False if not \
        present)",
    )
    parser.add_argument(
        "--elements_with_symbol",
        default=False,
        action="store_true",
        help="Use chemical elements symbol in the fixed candidate filter \
        (default False if not present)",
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
        "--output_path",
        type=str,
        default="./data/raw/lm-kbc/dataset/data/output.jsonl",
        help='Path to the output (default "./data/raw/lm-kbc/dataset/data\
        /output.jsonl")',
    )
    parser.add_argument(
        "--lm_candidates",
        type=str,
        default="bert-large-cased",
        help='LM to generate candidates (default "bert-large-cased")',
    )
    parser.add_argument(
        "--filter_before",
        default=False,
        action="store_true",
        help="Whether to filter out stop words before adding them in the candidate list \
        (default False if not present)",
    )
    parser.add_argument(
        "--is_baseline",
        default=False,
        action="store_true",
        help="Whether to use baseline (only language model withou entailment) \
        (default False if not present)",
    )
    parser.add_argument(
        "--lm_threshold",
        type=float,
        default=0.5,
        help="Language model threshold (default 0.5)",
    )
    parser.add_argument(
        "--calculate_lm_threshold",
        default=False,
        action="store_true",
        help="Whether to calculate LM threshold (This parameter regulates the ammount of \
        candidates per relation)  \
        (default False if not present)",
    )
    parser.add_argument(
        "--calculate_entailment_threshold",
        default=False,
        action="store_true",
        help="Whether to calculate only entailment threshold (This parameter regulates the \
                ammount of candidates per relation) (default False if not present)",
    )

    args = parser.parse_args()

    return args


def entailment(candidate_labels, return_scores, args, classifier, sents, row):
    subjectEntity = row["SubjectEntity"]
    relation = row["Relation"]
    template = f"{subjectEntity} {relation_to_prompt[relation]}"
    answers = {}
    if len(candidate_labels) > 0:
        if return_scores or type(candidate_labels[0]) == tuple:
            label2score = {label: score for (label, score) in candidate_labels}
            candidate_labels = [label for (label, score) in candidate_labels]

        res = classifier(
            sents,
            candidate_labels,
            hypothesis_template=template,
            multi_label=True,
        )
        for res_item in res:
            for label, score in zip(res_item["labels"], res_item["scores"]):
                if label in answers:
                    answers[label] += [score]
                else:
                    answers[label] = [score]

        answers = {
            label: mean([mean(score), label2score[label]])
            if return_scores
            else mean(score)
            for label, score in answers.items()
        }

    return answers


def baseline(args, unmasker):
    input_rows = read_lm_kbc_jsonl(args.input_path)
    predict_results = []
    for row in tqdm(input_rows):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        candidate_labels = get_candidates_from_lm_threshold(
            subjectEntity,
            relation,
            threshold=args.lm_threshold,
            return_scores=False,
            lm_candidates=unmasker,
            filter_before=True,
        )
        predict_results.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "ObjectEntities": candidate_labels,
            }
        )
    return predict_results


def baseline_fewshot(args, unmasker):
    input_rows = read_lm_kbc_jsonl(args.input_path)
    input_rows_dev_2 = read_lm_kbc_jsonl(args.input_path_dev_2)
    predict_results = []
    for row in tqdm(input_rows_dev_2):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        candidate_labels = get_candidates_from_lm_threshold(
            subjectEntity,
            relation,
            threshold=0.01,
            return_scores=True,
            lm_candidates=unmasker,
            filter_before=True,
        )
        predict_results.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "Candidates": candidate_labels,
            }
        )

    best_top_lm_threshold = {rel: None for rel in relation_to_prompt}
    best_top_lm_threshold_scores = {rel: None for rel in relation_to_prompt}

    begin = 0.01
    end = 0.99
    step = 0.01

    curr_t_top_lm = begin
    tidx_top_lm = 0

    while curr_t_top_lm < end:
        if tidx_top_lm % 10 == 0:
            print(f"tidx_top_lm: {tidx_top_lm};")

        for rel in relation_to_prompt:
            curr_allentry_results = []
            for res in predict_results:
                if res["Relation"] != rel:
                    continue
                curr_candidates = filter_candidates_with_lm_thres(
                    res["Candidates"], threshold=curr_t_top_lm, return_scores=None
                )
                curr_res = {
                    "SubjectEntity": res["SubjectEntity"],
                    "ObjectEntities": curr_candidates,
                    "Relation": res["Relation"],
                }
                curr_allentry_results.append(curr_res)

            scores_per_sr_pair = evaluate_per_sr_pair(
                curr_allentry_results, input_rows_dev_2
            )
            scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)
            if (
                best_top_lm_threshold_scores[rel] is None
                or scores_per_relation[rel]["f1"]
                > best_top_lm_threshold_scores[rel]["f1"]
            ):
                best_top_lm_threshold_scores[rel] = scores_per_relation[rel]
                best_top_lm_threshold[rel] = curr_t_top_lm

        curr_t_top_lm += step
        tidx_top_lm += 1

    best_candidates = []
    for row in tqdm(input_rows):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        candidate_labels = get_candidates_from_lm_threshold(
            subjectEntity,
            relation,
            threshold=best_top_lm_threshold[relation],
            return_scores=False,
            lm_candidates=unmasker,
            filter_before=True,
        )
        best_candidates.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "ObjectEntities": candidate_labels,
            }
        )

    return best_candidates


def calculate_lm_threshold(
    args, unmasker, return_scores, classifier, contexts, contexts_train
):
    print("**Calculating lm threshold and entailment threshold**")
    if args.use_candidates_fixed:
        fixed_candidates = get_fixed_candidates(
            expand_candidates=args.expand_candidates,
            elements_with_symbol=args.elements_with_symbol,
        )
        relation_to_fixed_candidates = {
            "CountryBordersWithCountry": fixed_candidates["countries"],
            "CountryOfficialLanguage": fixed_candidates["languages"],
            "ChemicalCompoundElement": fixed_candidates["elements"],
            "PersonLanguage": fixed_candidates["languages"],
            "PersonProfession": fixed_candidates["professions"],
            "PersonInstrument": fixed_candidates["instruments"],
            "PersonCauseOfDeath": fixed_candidates["causes_of_death"],
            "RiverBasinsCountry": fixed_candidates["countries"],
        }
    input_rows = read_lm_kbc_jsonl(args.input_path)
    input_rows_dev_2 = read_lm_kbc_jsonl(args.input_path_dev_2)
    predict_results = []
    for row in tqdm(input_rows_dev_2):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        sents = contexts_train[relation][subjectEntity]["contexts"][: args.k_sentences]
        candidate_labels = get_candidates_from_lm_threshold(
            subjectEntity,
            relation,
            threshold=0.01,
            return_scores=True,
            lm_candidates=unmasker,
            filter_before=args.filter_before,
        )
        if args.candidates_generation == "merge":
            candidate_labels_dict = {
                candidate[0]: candidate[1] for candidate in candidate_labels
            }
            if args.use_candidates_fixed and relation in relation_to_fixed_candidates:
                candidate_labels_dict = {
                    **candidate_labels_dict,
                    **{key: 1.0 for key in relation_to_fixed_candidates[relation]},
                }
            else:
                candidate_labels_dict = {
                    **candidate_labels_dict,
                    **{
                        key: 1.0
                        for key in get_candidates_from_contexts(
                            subjectEntity,
                            relation,
                            contexts_file=args.contexts_train_path,
                        )
                    },
                }
            candidate_labels = [
                (key, value) for key, value in candidate_labels_dict.items()
            ]

        if args.filter_fixed_candidates:
            candidate_labels = apply_fixed_candidates_filter(
                candidate_labels,
                subjectEntity,
                relation,
                args.contexts_train_path,
                return_scores=True,
            )

        answers = entailment(candidate_labels, False, args, classifier, sents, row)
        predict_results.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "Candidates": candidate_labels,
                "PredictAnswers": answers,
                "ObjectEntities": row["ObjectEntities"],
            }
        )

    best_top_lm_threshold = {rel: (None, None) for rel in relation_to_prompt}
    best_top_lm_threshold_scores = {rel: None for rel in relation_to_prompt}

    begin = 0.01
    end = 0.99
    step = 0.01

    curr_t_top_lm = begin
    tidx_top_lm = 0

    while curr_t_top_lm < end:
        if tidx_top_lm % 10 == 0:
            print(f"tidx_top_lm: {tidx_top_lm};")

        curr_t_threshold = begin
        tidx_threshold = 0

        while curr_t_threshold < end:
            # if tidx_threshold*100 % 10 == 0:
            #    print(f"tidx_threshold: {tidx_threshold};")
            for rel in relation_to_prompt:
                curr_allentry_results = []
                for res in predict_results:
                    if res["Relation"] != rel:
                        continue
                    curr_candidates = filter_candidates_with_lm_thres(
                        res["Candidates"], threshold=curr_t_top_lm, return_scores=None
                    )
                    # if args.filter_fixed_candidates:
                    #     curr_candidates = apply_fixed_candidates_filter(
                    #         curr_candidates,
                    #         res["SubjectEntity"],
                    #         res["Relation"],
                    #         args.contexts_path,
                    #     )
                    if len(res["PredictAnswers"]) == 0:
                        res["PredictAnswers"] = {}
                    curr_objs = set(
                        [
                            label
                            for label, score in res["PredictAnswers"].items()
                            if label in curr_candidates and score > curr_t_threshold
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
                    best_top_lm_threshold_scores[rel] is None
                    or scores_per_relation[rel]["f1"]
                    > best_top_lm_threshold_scores[rel]["f1"]
                ):
                    best_top_lm_threshold_scores[rel] = scores_per_relation[rel]
                    best_top_lm_threshold[rel] = (curr_t_top_lm, curr_t_threshold)
            curr_t_threshold += step
            tidx_threshold += 1

        curr_t_top_lm += step
        tidx_top_lm += 1

    print("**Best top_lm and threshold**:", best_top_lm_threshold)
    best_candidates = []
    for row in tqdm(input_rows):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        sents = contexts[relation][subjectEntity]["contexts"][: args.k_sentences]
        candidate_labels = get_candidates_from_lm_threshold(
            subjectEntity,
            relation,
            threshold=best_top_lm_threshold[relation][0],
            return_scores=return_scores,
            lm_candidates=unmasker,
            filter_before=args.filter_before,
        )
        if args.candidates_generation == "merge":
            candidate_labels = set(candidate_labels)
            if args.use_candidates_fixed and relation in relation_to_fixed_candidates:
                candidate_labels = candidate_labels.union(
                    set(relation_to_fixed_candidates[relation])
                )
            else:
                candidate_labels = candidate_labels.union(
                    set(
                        get_candidates_from_contexts(
                            subjectEntity, relation, contexts_file=args.contexts_path
                        )
                    )
                )

            candidate_labels = list(candidate_labels)

        if args.filter_fixed_candidates:
            candidate_labels = apply_fixed_candidates_filter(
                candidate_labels,
                subjectEntity,
                relation,
                args.contexts_path,
                return_scores=return_scores,
            )
        answers = entailment(
            candidate_labels, return_scores, args, classifier, sents, row
        )
        if len(answers) == 0:
            answers = {}
        curr_objs = set(
            [
                label
                for label, score in answers.items()
                if score > best_top_lm_threshold[relation][1]
            ]
        )
        best_candidates.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "ObjectEntities": list(curr_objs),
            }
        )
    print("**Using candidates with the best lm threshold and entailment threshold**")
    return best_candidates


def calculate_entailment_threshold(
    args, return_scores, classifier, contexts, contexts_train
):
    fixed_candidates = get_fixed_candidates(
        expand_candidates=args.expand_candidates,
        elements_with_symbol=args.elements_with_symbol,
    )
    relation_to_fixed_candidates = {
        "CountryBordersWithCountry": fixed_candidates["countries"],
        "CountryOfficialLanguage": fixed_candidates["languages"],
        "ChemicalCompoundElement": fixed_candidates["elements"],
        "PersonLanguage": fixed_candidates["languages"],
        "PersonProfession": fixed_candidates["professions"],
        "PersonInstrument": fixed_candidates["instruments"],
        "PersonCauseOfDeath": fixed_candidates["causes_of_death"],
        "RiverBasinsCountry": fixed_candidates["countries"],
    }
    print("**Calculating entailment threshold**")
    input_rows = read_lm_kbc_jsonl(args.input_path)
    input_rows_dev_2 = read_lm_kbc_jsonl(args.input_path_dev_2)
    predict_results = []
    for row in tqdm(input_rows_dev_2):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        sents = contexts_train[relation][subjectEntity]["contexts"][: args.k_sentences]
        if args.use_candidates_fixed and relation in relation_to_fixed_candidates:
            candidate_labels = relation_to_fixed_candidates[relation]
            if args.filter_fixed_candidates:
                candidate_labels = apply_fixed_candidates_filter(
                    candidate_labels, subjectEntity, relation, args.contexts_train_path
                )
        elif args.candidates_generation == "from_contexts":
            candidate_labels = get_candidates_from_contexts(
                subjectEntity, relation, contexts_file=args.contexts_train_path
            )
        answers = entailment(candidate_labels, False, args, classifier, sents, row)
        predict_results.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "Candidates": candidate_labels,
                "PredictAnswers": answers,
                "ObjectEntities": row["ObjectEntities"],
            }
        )

    best_entailment_threshold = {rel: None for rel in relation_to_prompt}
    best_entailment_threshold_scores = {rel: None for rel in relation_to_prompt}

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
                best_entailment_threshold_scores[rel] is None
                or scores_per_relation[rel]["f1"]
                > best_entailment_threshold_scores[rel]["f1"]
            ):
                best_entailment_threshold_scores[rel] = scores_per_relation[rel]
                best_entailment_threshold[rel] = curr_t_threshold
        curr_t_threshold += step
        tidx_threshold += 1

    print("**Best entailment threshold**:", best_entailment_threshold)
    best_candidates = []
    for row in tqdm(input_rows):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        sents = contexts[relation][subjectEntity]["contexts"][: args.k_sentences]
        if args.use_candidates_fixed and relation in relation_to_fixed_candidates:
            candidate_labels = relation_to_fixed_candidates[relation]
            if args.filter_fixed_candidates:
                candidate_labels = apply_fixed_candidates_filter(
                    candidate_labels, subjectEntity, relation, args.contexts_path
                )
        elif args.candidates_generation == "from_contexts":
            candidate_labels = get_candidates_from_contexts(
                subjectEntity, relation, contexts_file=args.contexts_path
            )
        answers = entailment(
            candidate_labels, return_scores, args, classifier, sents, row
        )
        if len(answers) == 0:
            answers = {}
        curr_objs = set(
            [
                label
                for label, score in answers.items()
                if score > best_entailment_threshold[relation]
            ]
        )
        best_candidates.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "ObjectEntities": list(curr_objs),
            }
        )
    print("**Using candidates with the best entailment threshold**")
    return best_candidates


def seek_and_entail(args, unmasker, threshold_entailment=0.5):
    fixed_candidates = get_fixed_candidates(
        expand_candidates=args.expand_candidates,
        elements_with_symbol=args.elements_with_symbol,
    )
    relation_to_fixed_candidates = {
        "CountryBordersWithCountry": fixed_candidates["countries"],
        "CountryOfficialLanguage": fixed_candidates["languages"],
        "ChemicalCompoundElement": fixed_candidates["elements"],
        "PersonLanguage": fixed_candidates["languages"],
        "PersonProfession": fixed_candidates["professions"],
        "PersonInstrument": fixed_candidates["instruments"],
        "PersonCauseOfDeath": fixed_candidates["causes_of_death"],
        "RiverBasinsCountry": fixed_candidates["countries"],
    }
    contexts = get_contexts_from_file(args.contexts_path)
    classifier = pipeline("zero-shot-classification", model=args.model, device=0)

    return_scores = False

    input_rows = read_lm_kbc_jsonl(args.input_path)
    predict_results = []
    for row in tqdm(input_rows):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        sents = contexts[relation][subjectEntity]["contexts"][: args.k_sentences]
        if args.use_candidates_fixed and relation in relation_to_fixed_candidates:
            candidate_labels = relation_to_fixed_candidates[relation]
            if args.filter_fixed_candidates:
                candidate_labels = apply_fixed_candidates_filter(
                    candidate_labels, subjectEntity, relation, args.contexts_path
                )
        elif args.candidates_generation == "from_contexts":
            candidate_labels = get_candidates_from_contexts(
                subjectEntity, relation, contexts_file=args.contexts_path
            )
        elif args.candidates_generation == "from_lm":
            candidate_labels = get_candidates_from_lm_threshold(
                subjectEntity,
                relation,
                threshold=args.lm_threshold,
                return_scores=False,
                lm_candidates=unmasker,
                filter_before=True,
                filter_punctuation=True,
            )
            if args.filter_fixed_candidates:
                candidate_labels = apply_fixed_candidates_filter(
                    candidate_labels,
                    subjectEntity,
                    relation,
                    args.contexts_path,
                    return_scores=return_scores,
                )
        elif args.candidates_generation == "merge":
            candidate_labels = set(
                get_candidates_from_lm_threshold(
                    subjectEntity,
                    relation,
                    threshold=args.lm_threshold,
                    return_scores=False,
                    lm_candidates=unmasker,
                    filter_before=True,
                )
            )
            if args.use_candidates_fixed and relation in relation_to_fixed_candidates:
                candidate_labels = candidate_labels.union(
                    set(relation_to_fixed_candidates[relation])
                )
            else:
                candidate_labels = candidate_labels.union(
                    set(
                        get_candidates_from_contexts(
                            subjectEntity, relation, contexts_file=args.contexts_path
                        )
                    )
                )
            candidate_labels = list(candidate_labels)
            if args.filter_fixed_candidates:
                candidate_labels = apply_fixed_candidates_filter(
                    candidate_labels,
                    subjectEntity,
                    relation,
                    args.contexts_path,
                    return_scores=return_scores,
                )

        answers = entailment(
            candidate_labels, return_scores, args, classifier, sents, row
        )

        filtered_answers = set(
            [label for label, score in answers.items() if score > threshold_entailment]
        )

        predict_results.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "ObjectEntities": list(filtered_answers),
            }
        )
    return predict_results


def save_results(args, results):
    with open(args.output_path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def main(args):
    if not args.calculate_entailment_threshold:
        unmasker = pipeline("fill-mask", model=args.lm_candidates, device=0)

    if args.is_baseline:
        if args.calculate_lm_threshold:
            print("Using baseline few shot...")
            results = baseline_fewshot(args, unmasker)
        else:
            print("Using baseline...")
            results = baseline(args, unmasker)
    elif args.calculate_entailment_threshold:
        contexts = get_contexts_from_file(args.contexts_path)
        contexts_train = get_contexts_from_file(args.contexts_train_path)
        classifier = pipeline("zero-shot-classification", model=args.model, device=0)
        return_scores = False
        results = calculate_entailment_threshold(
            args, return_scores, classifier, contexts, contexts_train
        )
    elif args.calculate_lm_threshold:
        contexts = get_contexts_from_file(args.contexts_path)
        contexts_train = get_contexts_from_file(args.contexts_train_path)
        classifier = pipeline("zero-shot-classification", model=args.model, device=0)
        return_scores = False
        results = calculate_lm_threshold(
            args, unmasker, return_scores, classifier, contexts, contexts_train
        )
    else:
        print("Seek and entailment...")
        results = seek_and_entail(args, unmasker)

    save_results(args, results)


if __name__ == "__main__":
    main(parse_args())
