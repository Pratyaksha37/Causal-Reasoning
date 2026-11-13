"""Real pretrained MiniLM encoding; no heuristic is labelled a transformer."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.data.brfss import file_sha256


class FrozenTransformerEncoder:
    """Frozen pretrained weights with attention-masked mean pooling and unit normalization."""

    def __init__(self, config: dict) -> None:
        path = Path(config["path"])
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(
                "Pretrained weights are unavailable. Run ./run.sh prepare-encoder first."
            )
        self.config = config
        provenance = json.loads((path / "provenance.json").read_text())
        if any(provenance.get(key) != config[key] for key in ["model_id", "revision"]):
            raise ValueError("Local encoder provenance does not match configured model/revision")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.metadata = {
            **provenance,
            "weights_sha256": file_sha256(path / "model.safetensors"),
            "architecture": type(self.model).__name__,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "encoder_finetuned": False,
            "pooling": "attention_mask_mean_then_l2_normalization",
            "device": "cpu",
            "max_length": config["max_length"],
        }

    def common_inputs(self, texts: list[str]) -> tuple[list[str], int]:
        """Apply the same tokenizer-bounded text to BOTH learned representations."""
        lengths = self.tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)[
            "input_ids"
        ]
        truncated = sum(len(tokens) > self.config["max_length"] for tokens in lengths)
        # Decode a bounded common representation so TF-IDF does not see an extra sentence tail.
        bounded = self.tokenizer(
            texts, truncation=True, max_length=self.config["max_length"], padding=False
        )["input_ids"]
        return self.tokenizer.batch_decode(bounded, skip_special_tokens=True), truncated

    def encode(self, texts: list[str], cache: Path | None = None) -> np.ndarray:
        """Encode without gradient updates; caches are keyed by weights AND exact texts."""
        key = hashlib.sha256(
            json.dumps([self.metadata, texts], sort_keys=True).encode()
        ).hexdigest()
        target = cache / f"{key}.npz" if cache is not None else None
        if target is not None and target.exists():
            values: np.ndarray = np.load(target, allow_pickle=False)["embeddings"]
            if (
                values.shape != (len(texts), self.model.config.hidden_size)
                or not np.isfinite(values).all()
            ):
                raise ValueError("Invalid embedding cache shape or values")
            return values
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.config["batch_size"]):
                encoded = self.tokenizer(
                    texts[start : start + self.config["batch_size"]],
                    padding=True,
                    truncation=True,
                    max_length=self.config["max_length"],
                    return_tensors="pt",
                )
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        values = np.concatenate(batches)
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, embeddings=values)
        return values


def prepare_encoder(config: dict, allow_download: bool = False) -> dict:
    """Copy an exact public checkpoint from HF cache; downloads require an explicit flag."""
    import shutil

    from huggingface_hub import snapshot_download

    target = Path(config["path"])
    target.mkdir(parents=True, exist_ok=True)
    names = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "README.md",
    ]
    snapshot = Path(
        snapshot_download(
            config["model_id"],
            revision=config["revision"],
            allow_patterns=names,
            local_files_only=not allow_download,
        )
    )
    for name in names:
        if (snapshot / name).exists():
            shutil.copyfile(snapshot / name, target / name)
    provenance = {key: config[key] for key in ["model_id", "revision"]}
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return {
        **provenance,
        "path": str(target),
        "weights_sha256": file_sha256(target / "model.safetensors"),
    }
