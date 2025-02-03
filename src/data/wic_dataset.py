from .base_dataset import BaseDataset

class WiCDataset(BaseDataset):
    def _format_input(self, item):
        return (
            f"Word: {item['word']}.\n"
            f"Sentence 1: {item['sentence1']}\n"
            f"Sentence 2: {item['sentence2']}"
        )