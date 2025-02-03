from .base_dataset import BaseDataset
import torch

class BoolQDataset(BaseDataset):
    def _format_input(self, item):
        """Format input text following BoolQ structure"""
        return (
            f"Question: {item['question']}\n"
            f"Passage: {item['passage']}"
        )
    
    def __getitem__(self, index):
        item = self.dataset_list[index]
        input_text = self._format_input(item)
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(1 if item["label"] else 0),
        }