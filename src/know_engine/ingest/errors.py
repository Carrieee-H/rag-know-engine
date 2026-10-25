class NotFoundError(LookupError):
    """Mapped to HTTP 404."""


class ConflictError(ValueError):
    """Business-rule or concurrency conflict; mapped to HTTP 409 with the message as plain text."""
