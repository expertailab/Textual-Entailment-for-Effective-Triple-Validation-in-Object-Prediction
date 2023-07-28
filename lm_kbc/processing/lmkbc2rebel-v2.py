import re
import json

from lm_kbc.common.utils import get_element_symbol, get_contexts_from_file
from lm_kbc.common.file_io import read_lm_kbc_jsonl

lmkbc_rel_map = {
    "ChemicalCompoundElement": "has part",
    "CompanyParentOrganization": "parent organization",
    "CountryBordersWithCountry": "shares border with",
    "CountryOfficialLanguage": "language used",
    "PersonCauseOfDeath": "cause of death",
    "PersonEmployer": "employer",
    "PersonInstrument": "instrument",
    "PersonLanguage": "language used",
    "PersonPlaceOfDeath": "place of death",
    "PersonProfession": "occupation",
    "RiverBasinsCountry": "basin country",
    "StateSharesBorderState": "shares border with",
}


def find_subject_objects(text, subject, objects, relation):
    lin_triplets = ""
    lower_text = text.lower()
    if subject.lower() in lower_text:
        objects_in_text = []
        for object_elem in objects:
            if (
                relation == "ChemicalCompoundElement"
                and re.search(get_element_symbol(object_elem), text)
            ) or object_elem.lower() in lower_text:
                objects_in_text.append(object_elem)
        if len(objects_in_text) > 0:
            lin_triplets += "<triplet> " + subject
            for object_elem in objects_in_text:
                lin_triplets += (
                    " <subj> " + object_elem + " <obj> " + lmkbc_rel_map[relation]
                )
    return lin_triplets


def convert_lmkbc2rebel(sample, contexts):
    my_list = []
    for item in sample:
        objects = [
            label for alternatives in item["ObjectEntities"] for label in alternatives
        ]
        for context in contexts[item["Relation"]][item["SubjectEntity"]]["contexts"][
            :10
        ]:
            triplets = find_subject_objects(
                context, item["SubjectEntity"], objects, item["Relation"]
            )
            if len(triplets) > 0:
                training_example = {
                    "title": "",
                    "context": context,
                    "triplets": triplets,
                }
                my_list.append({"id": str(id(training_example)), **training_example})
                break
    return my_list


def main():
    for dataset in ["train2", "dev2"]:
        for percentage in [5, 10, 20]:
            for k in range(10):
                print(f"{str(k)}/10")
                sample = read_lm_kbc_jsonl(
                    f"./data/processed/train/{dataset}-{str(percentage)}-{str(k)}.jsonl"
                )
                contexts = get_contexts_from_file(
                    "./data/processed/train/contexts/contexts_train.json"
                )
                my_list = convert_lmkbc2rebel(sample, contexts)
                with open(
                    "".join(
                        [
                            "data/processed/train/lm_kbc_{dataset}-v2_rebel_",
                            f"{str(percentage)}-{str(k)}.json",
                        ]
                    ),
                    "a",
                ) as f:
                    for result in my_list:
                        f.write(json.dumps(result) + "\n")

        sample = read_lm_kbc_jsonl("./data/processed/train/{dataset}.jsonl")
        contexts = get_contexts_from_file(
            "./data/processed/train/contexts/contexts_train.json"
        )
        my_list = convert_lmkbc2rebel(sample, contexts)
        with open("data/processed/train/lm_kbc_{dataset}-v2_rebel.json", "a") as f:
            for result in my_list:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
