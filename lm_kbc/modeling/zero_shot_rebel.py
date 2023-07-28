import argparse

from tqdm import tqdm
from transformers import pipeline

from lm_kbc.common.utils import read_lm_kbc_jsonl, get_contexts_from_file
from lm_kbc.modeling.zero_shot_entailment import save_results


def parse_args():
    parser = argparse.ArgumentParser(description="REBEL baseline for LM-KBC dataset.")
    parser.add_argument(
        "--model",
        metavar="MODEL",
        type=str,
        default="Babelscape/rebel-large",
        help='Model to use relation extraction \
                (default "Babelscape/rebel-large")',
    )
    parser.add_argument(
        "--contexts_path",
        type=str,
        default="./data/processed/dev/contexts/contexts.json",
        help='Path to the contexts (default "./data/processed/dev/contexts\
        /contexts.json")',
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./data/raw/lm-kbc/dataset/data/dev.jsonl",
        help='Path to the input (default "./data/raw/lm-kbc/dataset/data/dev.jsonl")',
    )
    parser.add_argument(
        "--k_sentences",
        metavar="k",
        type=int,
        default=3,
        help="Number of contexts to use for question answering (3 by default)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/raw/lm-kbc/dataset/data/output.jsonl",
        help='Path to the output (default "./data/raw/lm-kbc/dataset/data\
        /output.jsonl")',
    )

    args = parser.parse_args()

    return args


lmkbc_rel_map = {
    "ChemicalCompoundElement": ["has part"],
    "CompanyParentOrganization": ["parent organization"],
    "CountryBordersWithCountry": ["shares border with"],
    "CountryOfficialLanguage": ["language used"],
    "PersonCauseOfDeath": ["cause of death"],
    "PersonEmployer": ["employer"],
    "PersonInstrument": ["instrument"],
    "PersonLanguage": ["language used"],
    "PersonPlaceOfDeath": ["place of death"],
    "PersonProfession": ["occupation"],
    "RiverBasinsCountry": ["basin country"],
    "StateSharesBorderState": ["shares border with"],
}


# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )
    return triplets


def relation_extraction(args, triplet_extractor, sents, row):
    subjectEntity = row["SubjectEntity"]
    relation = row["Relation"]
    answers = {}
    # sents = [relation + " " + sent for sent in sents]
    if len(sents) > 0:
        extracted_text = triplet_extractor.tokenizer.batch_decode(
            [
                elem["generated_token_ids"]
                for elem in triplet_extractor(
                    sents, return_tensors=True, return_text=False
                )
            ]
        )
        # print(extracted_text)
        extracted_triplets = [
            extract_triplets(extracted_text_elem)
            for extracted_text_elem in extracted_text
        ]
        # print(extracted_triplets)
        # exit()
        answers = {
            answer["tail"].lower()
            for extracted_triplets_elem in extracted_triplets
            for answer in extracted_triplets_elem
            if (
                answer["head"].lower() == subjectEntity.lower()
                and answer["type"] in lmkbc_rel_map[relation]
            )
        }
        # print(answers)
    return list(answers)


def main(args):
    # model_name = "Babelscape/rebel-large"
    # model_name = "/home/cristian.berrio@EXPERT.AI/rebel/model/rebel-large-5-0"
    triplet_extractor = pipeline(
        "text2text-generation", model=args.model, tokenizer=args.model, device=0
    )
    contexts = get_contexts_from_file(args.contexts_path)
    input_rows = read_lm_kbc_jsonl(args.input_path)
    predict_results = []
    for row in tqdm(input_rows):
        subjectEntity = row["SubjectEntity"]
        relation = row["Relation"]
        # if relation != "StateSharesBorderState":
        #    continue
        # subjectEntity = ""
        sents = contexts[relation][subjectEntity]["contexts"][: args.k_sentences]
        # print(sents)
        answers = relation_extraction(args, triplet_extractor, sents, row)
        predict_results.append(
            {
                "SubjectEntity": subjectEntity,
                "Relation": relation,
                "ObjectEntities": answers,
            }
        )
    # print(predict_results)
    save_results(args, predict_results)


if __name__ == "__main__":
    main(parse_args())
