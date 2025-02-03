from .base_dataset import BaseDataset
import torch

class COPADataset(BaseDataset):
    def _format_input(self, item):
        """Format input text following COPA structure"""
        return (
            f"Premise: {item['premise']}\n"
            f"Question: {item['question']}\n"
            f"Choice 1: {item['choice1']}\n"
            f"Choice 2: {item['choice2']}"
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
            "label": torch.tensor(item["label"]),
        }