from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "itsGoodman/deberta-heading-detectorert-tiny-heading-classifier-v1"  # <-- your HF repo name
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
