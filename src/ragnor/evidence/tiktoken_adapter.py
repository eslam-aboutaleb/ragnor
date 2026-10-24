"""Optional tiktoken model-token counter, loaded only when requested."""

from __future__ import annotations


class TiktokenTokenizer:
    """Count exact BPE tokens for the selected tiktoken encoding.

    A chosen encoding must match the provider's actual model/revision.
    This only counts the given string, not host prompt serialization.
    """

    def __init__(self, *, encoding_name: str | None = None, model: str | None = None) -> None:
        if (encoding_name is None) == (model is None):
            raise ValueError("specify exactly one of encoding_name or model")
        try:
            import tiktoken  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "tiktoken extra not installed: pip install 'ragnor[tokenizers-tiktoken]'"
            ) from exc

        try:
            encoding = (
                tiktoken.get_encoding(encoding_name)
                if encoding_name is not None
                else tiktoken.encoding_for_model(model)
            )
        except KeyError as exc:
            raise ValueError("unknown tiktoken model or encoding") from exc
        self._encoding = encoding

    @property
    def is_exact(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"tiktoken:{self._encoding.name}"

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))
