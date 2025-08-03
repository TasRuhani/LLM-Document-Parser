# clause_classifier.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

class ClauseClassifier:
    def __init__(self, model_path="./model/legal-bert-finetuned-context", threshold=0.5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load config and set number of labels explicitly
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 41  # Hardcoded to match your trained model
        
        # Load model and force it to CPU to avoid meta tensor issues
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        self.model.eval()
        self.model.to("cpu")

        self.threshold = threshold
        self.label_list = list(self.model.config.id2label.values())

    def predict(self, clause_text):
        # Tokenize and move inputs to CPU
        inputs = self.tokenizer(clause_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Filter predictions based on threshold
        predictions = [
            (self.label_list[i], float(probs[i]))
            for i in range(len(probs))
            if probs[i] > self.threshold
        ]
        return predictions
