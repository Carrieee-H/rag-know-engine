from enum import StrEnum


class DocumentStatus(StrEnum):
    UPLOADED = "UPLOADED"
    CONVERTING = "CONVERTING"
    CONVERTED = "CONVERTED"
    CHUNKED = "CHUNKED"
    VECTOR_STORED = "VECTOR_STORED"
    FAILED = "FAILED"  # new: terminal state after the worker exhausts its retries


class SegmentStatus(StrEnum):
    STORED = "STORED"
    VECTOR_STORED = "VECTOR_STORED"


class SplitType(StrEnum):
    LENGTH = "LENGTH"
    TITLE = "TITLE"
    REGEX = "REGEX"
    SMART = "SMART"
    SEPARATOR = "SEPARATOR"


class MessageType(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class RetrievalSource(StrEnum):
    VECTOR = "VECTOR"
    KEYWORD = "KEYWORD"
    BOTH = "BOTH"


class EvalRunStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
