import logging
import os
from pathlib import Path

import numpy as np
import sentencepiece

class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48, *, strict: bool = False):
        self._max_len = max_len
        self._strict = strict

        default_path = Path(__file__).resolve().parents[3] / "assets/tokenizer/paligemma_tokenizer.model"
        path = Path(os.environ.get("USV_TOKENIZER_PATH", str(default_path))).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"PaliGemma tokenizer not found: {path}. "
                "Keep assets/tokenizer/paligemma_tokenizer.model with this project "
                "or set USV_TOKENIZER_PATH to an existing local file."
            )
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            if self._strict and (not np.isfinite(state).all() or np.any(np.abs(state) > 1)):
                raise ValueError("State must be finite and normalized to [-1, 1].")
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            if self._strict:
                discretized_state = np.clip(discretized_state, 0, 255)
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                if self._strict:
                    raise ValueError(f"Task/state prompt has {len(tokens)} tokens, limit is {self._max_len}.")
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
