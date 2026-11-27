"""Test helper: make llama-index-core's bundled NLTK data loadable.

Recent NLTK refuses to read data from a directory that other accounts can write
to. CI runners install packages with a permissive umask, so the bundled
``_static/nltk_cache`` directory trips that check and ``SentenceSplitter``
fails with ``LookupError: Resource 'stopwords' not found``. Removing the
group/other write bits is exactly what NLTK's error message asks for. The
runner's package cache can also hardlink the data files, and the path-security
layer refuses multiply-linked files, so those are replaced by plain copies.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def harden_nltk_cache() -> None:
    try:
        import llama_index.core  # noqa: PLC0415
    except ImportError:
        return
    cache = Path(llama_index.core.__file__).parent / "_static" / "nltk_cache"
    if not cache.exists():
        return
    for root, _dirs, files in os.walk(cache):
        for name in files:
            _unlink_shared(Path(root) / name)
    for root, dirs, files in os.walk(cache):
        for name in [root, *[os.path.join(root, n) for n in dirs + files]]:
            mode = os.stat(name).st_mode
            if mode & (stat.S_IWGRP | stat.S_IWOTH):
                os.chmod(name, mode & ~(stat.S_IWGRP | stat.S_IWOTH))


def _unlink_shared(path: Path) -> None:
    """Replace a hardlinked file with an independent copy of the same bytes."""
    if path.is_symlink() or os.stat(path).st_nlink <= 1:
        return
    data = path.read_bytes()
    mode = stat.S_IMODE(os.stat(path).st_mode)
    path.unlink()
    path.write_bytes(data)
    path.chmod(mode)
