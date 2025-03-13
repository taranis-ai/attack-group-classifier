import json
import torch
from pathlib import Path
from attack_group_classifier.config import Config
from attack_group_classifier.predictor import Predictor
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Mpnet(Predictor):
    model_name = "selfconstruct3d/AttackGroup-MPNET"
    label_file_path = Path(__file__).parent / "label_to_groupid.json"

    def __init__(self):
        with open(self.label_file_path, "r") as f:
            self.label_to_groupid = json.load(f)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=len(self.label_to_groupid))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def predict(self, text: str) -> str:
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=Config.MAX_TEXT_LENGTH, return_tensors="pt")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

        return self.label_to_groupid[str(predicted_label)]
