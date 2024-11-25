import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@lru_cache(maxsize=None)
def load_huggingface_model(
        model_name_or_path: str = 'BAAI/bge-reranker-base',
        device: str = 'cuda'):

    logging.info(f"Loading huggingface model {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path)
    model.eval()
    model.to(device)
    logging.info(f"Model {model_name_or_path} loaded.")
    return model, tokenizer

@dataclass
class Reranker:
    """Changing a rerank model requires a comprehensive testing and evalution process.
    You need to ensure that the metrics like MRR, NGDC, ACC3, ACC5, etc. are improved or at least not degraded.
    Conclusion: If there is already a deployed model, do not change it in any case.
    """

    model_name_or_path: str = 'BAAI/bge-reranker-base'

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, self.tokenizer = load_huggingface_model(
            model_name_or_path=self.model_name_or_path,
            device=self.device)

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        print(f"Reranking {len(documents)} documents...")
        start = time.time()
        pairs = [(str(query), str(doc)) for doc in documents]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512)

            inputs.to(self.device)
            scores = self.model(
                **inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu().numpy().astype('float')
        end = time.time()
        print(f"Reranking taking {end-start:.2f} seconds")
        return scores


if __name__ == '__main__':
    reranker = Reranker()
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France.",
        "Beijing is the capital of China.",
        "London is the capital of UK."
    ]
    print(reranker.rerank(query, documents))