from .base_dataset import BaseDataset
import torch

class MultiRCDataset(BaseDataset):
    def __init__(self, dataset_list, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._process_dataset(dataset_list)

    def _process_dataset(self, dataset_list):
        """Convert complex passage/question/answer structure into flat samples"""
        samples = []
        for item in dataset_list:
            passage = item["passage"]["text"]
            for question_data in item["passage"]["questions"]:
                question = question_data["question"]
                for answer_data in question_data["answers"]:
                    samples.append({
                        "passage": passage,
                        "question": question,
                        "answer": answer_data["text"],
                        "label": answer_data["label"]
                    })
        return samples

    def _format_input(self, item):
        """Format input text following MultiRC structure"""
        return (
            f"Passage: {item['passage']}\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )
    
    def __getitem__(self, index):
        item = self.samples[index]
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
            "label": torch.tensor(item["label"], dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)