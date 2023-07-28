from lm_kbc.common.file_io import read_lm_kbc_jsonl
from sklearn.model_selection import train_test_split
import json

relations = {
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

sample = read_lm_kbc_jsonl(f"./data/raw/lm-kbc/dataset/data/train.jsonl")

for item in sample:
    relations[item['Relation']].append(item)

train2_list = []
dev2_list = []
for relation in relations:
    train2, dev2 = train_test_split(relations[relation],test_size=0.2, random_state=42)
    train2_list += train2
    dev2_list += dev2

with open(f"data/processed/train/train2.jsonl", "a") as f:
    for result in train2_list:
        f.write(json.dumps(result) + "\n")

with open(f"data/processed/train/dev2.jsonl", "a") as f:
    for result in dev2_list:
        f.write(json.dumps(result) + "\n")

