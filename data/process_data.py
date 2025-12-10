from transformers import AutoTokenizer


class Tokenizer():
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
