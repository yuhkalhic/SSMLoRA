from .base_dataset import BaseDataset

class WSCDataset(BaseDataset):
    def _format_input(self, item):
        return (
            f"Context: {item['text']}\n"
            f"Span 1: {item['target']['span1_text']}\n"
            f"Span 2: {item['target']['span2_text']}"
        )