"""The installed legacy API must reject wrong argument types, not become Any."""

from ragnor.config import RagnorConfig
from ragnor.types import Document

wrong_document = Document(id=123, text=False)
wrong_configuration = RagnorConfig(embedding_model=123)
