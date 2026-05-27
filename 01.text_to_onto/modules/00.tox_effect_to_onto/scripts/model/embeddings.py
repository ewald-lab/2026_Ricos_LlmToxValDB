import logging
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if self.model is not None:
            return

        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            # Handle specific model paths if needed, or use default HF
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side='left')
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise e

    def get_embedding(self, text: str) -> bytes:
        """Generates embedding for a single string and returns bytes."""
        if self.model is None:
            self.load()

        # Some models (like Llama/Qwen) benefit from a prompt prefix for query embedding
        # but for now we follow the standard approach.
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8192)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use last token pooling as requested for these decoder-only models
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            vector = embeddings[0].float().cpu().numpy().astype(np.float32).tobytes()

        return vector