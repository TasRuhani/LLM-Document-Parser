import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

class ClauseClassifier:
    def __init__(self, model_path="./model/legal-bert-finetuned", threshold=0.5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 41 
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        self.model.eval()
        self.model.to("cpu")

        self.threshold = threshold
        self.label_list = list(self.model.config.id2label.values())

    def predict(self, clause_text):
        inputs = self.tokenizer(clause_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        predictions = [
            (self.label_list[i], float(probs[i]))
            for i in range(len(probs))
            if probs[i] > self.threshold
        ]
        return predictions
