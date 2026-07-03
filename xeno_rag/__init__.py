"""Xeno Series Wiki RAG chatbot package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xeno-rag")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+source"
