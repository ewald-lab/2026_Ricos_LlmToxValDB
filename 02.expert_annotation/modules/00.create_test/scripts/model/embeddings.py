import logging
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import numpy as np

logger = logging.getLogger(__name__)

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Description:
    Performs last token pooling on hidden states using the attention mask.

    Args:
        last_hidden_states (Tensor): The hidden states from the model.
        attention_mask (Tensor): The attention mask tensor.

    Returns:
        Tensor: The pooled hidden state representation.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        """
        Description:
        Initializes the EmbeddingModel wrapper class.

        Args:
            model_name (str): Name or path of the pretrained model to load.

        Returns:
            None
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """
        Description:
        Loads the embedding model and tokenizer into memory if not already loaded.

        Args:
            None

        Returns:
            None
        """
        if self.model is not None:
            return

        logger.info(f"Loading embedding model: {self.model_name}")
        try:
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
        """
        Description:
        Generates embedding for a single string and returns it as a bytes object.

        Args:
            text (str): The input text to encode.

        Returns:
            bytes: The binary representation of the float32 embedding vector.
        """
        if self.model is None:
            self.load()

       
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8192)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            vector = embeddings[0].float().cpu().numpy().astype(np.float32).tobytes()

        return vector