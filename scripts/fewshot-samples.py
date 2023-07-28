from lm_kbc.common.file_io import read_lm_kbc_jsonl
import json
import random


for dataset in ["train2","dev2"]:
    random.seed(42)

    relation_to_prompt = {
        "CountryBordersWithCountry": [],
        "CountryOfficialLanguage": [],
        "StateSharesBorderState": [],
        "RiverBasinsCountry": [],
        "ChemicalCompoundElement": [],
        "PersonLanguage": [],
        "PersonProfession": [],
        "PersonInstrument": [],
        "PersonEmployer": [],
        "PersonPlaceOfDeath": [],
        "PersonCauseOfDeath": [],
        "CompanyParentOrganization": [],
    }
    input_rows = read_lm_kbc_jsonl(f'./data/processed/train/{dataset}.jsonl')
    for row in input_rows:
        relation_to_prompt[row['Relation']].append(row)
    
    for percentage in [5,10,20]:
    
        for i in range(10):
            sample = [elem for relation in relation_to_prompt for elem in random.sample(relation_to_prompt[relation],int(len(relation_to_prompt[relation])*percentage*0.01))]
            with open(f"./data/processed/train/{dataset}-{str(percentage)}-{str(i)}.jsonl", "a") as f:
                for result in sample:
                    f.write(json.dumps(result) + "\n")
