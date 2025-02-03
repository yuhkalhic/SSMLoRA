from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dataset_list, tokenizer, max_length=100):
        self.dataset_list = dataset_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset_list)

    def _format_input(self, item):
        """Format input text based on dataset requirements"""
        raise NotImplementedError

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
            "label": torch.tensor(1 if item["label"] else 0)
        }