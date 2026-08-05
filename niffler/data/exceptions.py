"""Typed exceptions for the ``niffler.data`` package.

Downloaders used to swallow every failure and ``return None``, which made a
broken network call indistinguishable from "the venue genuinely has no candles
for this range" -- and both were trivially ignored by callers. The hierarchy
below makes that distinction explicit and forces callers to handle it.

Hierarchy::

    NifflerDataError
    +-- DownloadError            the download broke (transport/venue/parsing)
    +-- NoDataAvailableError     the request succeeded, the venue had no data
    +-- InvalidTimeframeError    caller asked for an unsupported timeframe
"""

from typing import Optional


class NifflerDataError(Exception):
    """Base class for every error raised by the ``niffler.data`` package."""


class DownloadError(NifflerDataError):
    """Raised when a download genuinely failed.

    This means the data could not be retrieved: a network error, an exchange
    rejecting the request, an unusable response payload, or a missing client
    library. It never means "the venue has no data" -- that is
    :class:`NoDataAvailableError`.

    Args:
        message: Human readable description of what broke.
        source: Optional identifier of the data source (exchange id, provider).
        symbol: Optional symbol/ticker the download was for.
    """

    def __init__(self, message: str, source: Optional[str] = None,
                 symbol: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.source = source
        self.symbol = symbol


class NoDataAvailableError(NifflerDataError):
    """Raised when the request succeeded but returned no rows.

    This is a normal, expected outcome (delisted ticker, market holiday range,
    a symbol that simply has no history at that resolution) and is deliberately
    NOT a subclass of :class:`DownloadError` so callers can treat it separately.

    Args:
        message: Human readable description.
        source: Optional identifier of the data source.
        symbol: Optional symbol/ticker the download was for.
    """

    def __init__(self, message: str, source: Optional[str] = None,
                 symbol: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.source = source
        self.symbol = symbol


class InvalidTimeframeError(NifflerDataError):
    """Raised when a caller requests a timeframe the downloader cannot serve.

    Args:
        timeframe: The rejected timeframe.
        supported: The timeframes the downloader does support.
        source: Optional identifier of the data source.
    """

    def __init__(self, timeframe: str, supported: list, source: Optional[str] = None) -> None:
        super().__init__(
            f"Invalid timeframe '{timeframe}'"
            + (f" for {source}" if source else "")
            + f". Supported: {supported}"
        )
        self.timeframe = timeframe
        self.supported = supported
        self.source = source
