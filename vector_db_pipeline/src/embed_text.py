"""
embed_text.py — Text embedding using Qwen3-Embedding.

Uses INT8 quantization via bitsandbytes when CUDA is available.
Falls back to float16 on MPS and float32 on CPU.
"""

import torch
from transformers import AutoModel, AutoTokenizer


class TextEmbedder:
    """Wraps Qwen3-Embedding for text vector generation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str | None = None,
        quantization: str | None = None,
    ):
        self.device = device or self._auto_device()
        print(f"[TextEmbedder] Loading {model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if quantization == "int8" and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quant_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
            print("[TextEmbedder] INT8 quantization enabled.")
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)

        self.model.eval()
        print(f"[TextEmbedder] Ready. Output dim: {self.model.config.hidden_size}")

    def _auto_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        return self.embed_texts([text])[0]

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of normalized vectors."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def _mean_pooling(self, outputs, attention_mask) -> torch.Tensor:
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def free(self):
        """Release GPU/MPS memory."""
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
