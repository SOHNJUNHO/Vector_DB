"""
embed_visual.py — Visual embedding using Qwen3-VL-Embedding-2B.

Accepts both images (PNG paths) and text strings, producing
vectors in a shared multimodal embedding space.
"""

import torch


class VisualEmbedder:
    """
    Wraps Qwen3-VL-Embedding for unified image+text embedding.
    Output vectors are L2-normalized to unit length.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str | None = None,
    ):
        self.device = device or self._auto_device()
        self.model_name = model_name

        print(f"[VisualEmbedder] Loading {model_name} on {self.device}...")

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        cfg = self.model.config
        dim = (
            getattr(cfg, "hidden_size", None)
            or getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            or "?"
        )
        print(f"[VisualEmbedder] Ready. Output dim: {dim}")

    def _auto_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @torch.no_grad()
    def embed_image(self, image_path: str) -> list[float]:
        """Embed a single image file. Returns a normalized list of floats."""
        return self._process([{"image": image_path}])[0]

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """Embed a text string. Returns a normalized list of floats."""
        return self._process([{"text": text}])[0]

    @torch.no_grad()
    def embed_batch(
        self,
        image_paths: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> list[list[float]]:
        """Embed a batch of images and/or texts (images first, then texts)."""
        inputs = []
        if image_paths:
            inputs.extend([{"image": p} for p in image_paths])
        if texts:
            inputs.extend([{"text": t} for t in texts])
        if not inputs:
            return []
        return self._process(inputs)

    def _process(self, inputs: list[dict]) -> list[list[float]]:
        """Core embedding: encode → L2-normalize → list of floats."""
        if hasattr(self.model, "encode"):
            embeddings = self.model.encode(
                inputs,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        else:
            embeddings = self._manual_encode(inputs)

        # L2-normalize so vectors are unit-length (matches COSINE distance in Qdrant)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        if embeddings.ndim == 1:
            return [embeddings.cpu().tolist()]
        return embeddings.cpu().tolist()

    def _manual_encode(self, inputs: list[dict]) -> torch.Tensor:
        """Fallback: manual tokenization + forward pass + mean pooling."""
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
                # Qwen3-VL processor requires image tokens embedded via chat template
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": ""},
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch = processor(
                    images=[img], text=[text], return_tensors="pt", padding=True
                ).to(self.device)
                out = self.model(**batch)
            else:
                text = inp.get("text", "")
                batch = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                out = self.model(**batch)

            if hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state.mean(dim=1)
            elif hasattr(out, "embedding"):
                emb = out.embedding.unsqueeze(0)
            else:
                emb = out[0].mean(dim=1) if isinstance(out, tuple) else out.mean(dim=1)

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
