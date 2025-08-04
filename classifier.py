from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch

class ClauseClassifier:
    def __init__(self, model_path="./model/legal-bert-finetuned", threshold=0.3, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 41
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
        self.model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.label_list = list(self.model.config.id2label.values())
        self.device = device

    def predict(self, texts, batch_size=32):
        predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()
            for prob in probs:
                predictions.append([(self.label_list[j], float(prob[j])) for j in range(len(prob)) if prob[j] > self.threshold])
        return predictions
