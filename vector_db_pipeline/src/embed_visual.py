"""
embed_visual.py — Visual embedding using Qwen3-VL-Embedding-2B.

Accepts both images (PNG paths) and text strings, producing
vectors in a shared multimodal embedding space.

Uses the official Qwen3VLEmbedder API.
"""

import os
import sys

import torch

# Add project root so we can import the official helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class VisualEmbedder:
    """
    Wraps Qwen3-VL-Embedding for unified image+text embedding.
    Uses the official Qwen3VLEmbedder from the model repo.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str | None = None,
    ):
        self.device = device or self._auto_device()
        self.model_name = model_name

        print(f"[VisualEmbedder] Loading {model_name} on {self.device}...")

        # Lazy import — official embedder is in the model repo
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        print(f"[VisualEmbedder] Ready. Output dim: {self.model.config.hidden_size}")

    def _auto_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.no_grad()
    def embed_image(self, image_path: str) -> list[float]:
        """Embed a single image file. Returns a list of floats."""
        inputs = [{"image": image_path}]
        result = self._process(inputs)
        return result[0]

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """Embed a text string. Returns a list of floats."""
        inputs = [{"text": text}]
        result = self._process(inputs)
        return result[0]

    @torch.no_grad()
    def embed_batch(
        self,
        image_paths: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> list[list[float]]:
        """
        Embed a batch of images and/or texts.

        Args:
            image_paths: List of image file paths (or None)
            texts: List of text strings (or None)

        Returns:
            Embeddings in order: all image_paths embeddings first, then all
            texts embeddings. The two groups are not interleaved.
        """
        inputs = []
        if image_paths:
            inputs.extend([{"image": p} for p in image_paths])
        if texts:
            inputs.extend([{"text": t} for t in texts])

        if not inputs:
            return []

        return self._process(inputs)

    def _process(self, inputs: list[dict]) -> list[list[float]]:
        """
        Core embedding via the model's encode method.
        """
        # Try the official encode API first
        if hasattr(self.model, "encode"):
            embeddings = self.model.encode(
                inputs,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        else:
            # Fallback: manual tokenization + forward pass
            embeddings = self._manual_encode(inputs)

        # Convert to list of lists
        if embeddings.ndim == 1:
            return [embeddings.cpu().tolist()]
        return embeddings.cpu().tolist()

    def _manual_encode(self, inputs: list[dict]) -> torch.Tensor:
        """Fallback manual encoding if encode() is not available."""
        all_embeddings = []
        processor = None
        for inp in inputs:
            if "image" in inp:
                if processor is None:
                    from transformers import AutoProcessor
                    processor = AutoProcessor.from_pretrained(
                        self.model_name, trust_remote_code=True
                    )
                from PIL import Image
                img = Image.open(inp["image"]).convert("RGB")
                batch = processor(images=img, return_tensors="pt").to(self.device)
                out = self.model(**batch)
            else:
                # Text input
                text = inp.get("text", "")
                batch = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                out = self.model(**batch)

            # Pool
            if hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state.mean(dim=1)
            elif hasattr(out, "embedding"):
                emb = out.embedding.unsqueeze(0)
            else:
                emb = out[0] if isinstance(out, tuple) else out

            all_embeddings.append(emb.squeeze(0))

        return torch.stack(all_embeddings)

    def free(self):
        """Release GPU memory."""
        del self.model
        del self.tokenizer
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
