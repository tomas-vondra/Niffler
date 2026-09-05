"""The single registry of data sources addressable by name.

Adding a source to Niffler is **one entry in :data:`DOWNLOAD_SOURCES`**.
``scripts/download_data.py`` used to hardcode ``choices=['ccxt', 'yahoo']`` and then
branch on ``args.source`` twice more - once to build the output filename, once to
construct the downloader and translate the CLI arguments into its ``download()``
call. Three coordinated edits, and a forgotten one either crashed or, worse, wrote
a file named after the wrong source.

The two shipped downloaders genuinely disagree about their ``download()``
signatures - CCXT wants an exchange id and millisecond epochs, Yahoo Finance wants
a ticker and date strings - so a registry entry carries the translation from a
source-neutral :class:`DownloadRequest` alongside the class. That is still one
registration: everything a source needs lives in its own entry.

The options a source accepts are read off ``download()`` with
:func:`inspect.signature`, the same technique
:func:`niffler.strategies.registry.get_strategy_parameter_names` uses, so an option
the chosen source does not accept (``--exchange`` on ``yahoo``) is an error naming
what it does accept, rather than being silently ignored.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type

import pandas as pd

from .base_downloader import BaseDownloader
from .ccxt_downloader import CCXTDownloader
from .yahoo_finance_downloader import YahooFinanceDownloader


@dataclass(frozen=True)
class DownloadRequest:
    """What the CLI was asked to download, in source-neutral terms.

    ``start_date`` / ``end_date`` keep the raw strings the user typed as well as the
    parsed timestamps: a source that speaks dates is handed exactly what was typed,
    so reformatting can never change what a venue is asked for.

    Attributes:
        source: Registered source name
        symbol: Trading pair or ticker, as typed
        timeframe: Timeframe/interval, as typed
        start_date: Raw start date string
        end_date: Raw end date string
        start: Parsed start timestamp
        end: Parsed end timestamp
        options: Source-specific options, defaults already applied
    """

    source: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    start: pd.Timestamp
    end: pd.Timestamp
    options: Dict[str, Any] = field(default_factory=dict)


def _default_file_tag(request: DownloadRequest) -> str:
    """Name the source itself in the output filename."""
    return request.source


def _default_describe(request: DownloadRequest) -> str:
    """Describe the download for the CLI's error messages."""
    return f"{request.symbol} from {request.source}"


@dataclass(frozen=True)
class DownloadSource:
    """One registered data source: the downloader plus its CLI translation.

    Attributes:
        downloader_class: The :class:`BaseDownloader` subclass to construct
        build_download_kwargs: Turns a request into that downloader's ``download()``
            keyword arguments
        option_defaults: Source options applied when the user passed none
        file_tag: The ``{SOURCE}`` component of the default output filename
        describe: Human-readable description used in the CLI's error messages
    """

    downloader_class: Type[BaseDownloader]
    build_download_kwargs: Callable[[DownloadRequest], Dict[str, Any]]
    option_defaults: Dict[str, Any] = field(default_factory=dict)
    file_tag: Callable[[DownloadRequest], str] = _default_file_tag
    describe: Callable[[DownloadRequest], str] = _default_describe


def _ccxt_download_kwargs(request: DownloadRequest) -> Dict[str, Any]:
    """CCXT takes an exchange id and millisecond epochs."""
    return {
        'exchange_id': request.options['exchange_id'],
        'symbol': request.symbol,
        'timeframe': request.timeframe,
        'start_ms': int(request.start.timestamp() * 1000),
        'end_ms': int(request.end.timestamp() * 1000),
    }


def _yahoo_download_kwargs(request: DownloadRequest) -> Dict[str, Any]:
    """Yahoo Finance takes a ticker and the date strings as typed."""
    return {
        'ticker': request.symbol,
        'start_date': request.start_date,
        'end_date': request.end_date,
        'interval': request.timeframe,
    }


# The registry. Adding a source means adding an entry here - nothing else.
DOWNLOAD_SOURCES: Dict[str, DownloadSource] = {
    'ccxt': DownloadSource(
        downloader_class=CCXTDownloader,
        build_download_kwargs=_ccxt_download_kwargs,
        option_defaults={'exchange_id': 'binance'},
        # The file is named after the venue, not after "ccxt".
        file_tag=lambda request: request.options['exchange_id'],
        describe=lambda request: f"{request.symbol} from {request.options['exchange_id']}",
    ),
    'yahoo': DownloadSource(
        downloader_class=YahooFinanceDownloader,
        build_download_kwargs=_yahoo_download_kwargs,
        describe=lambda request: f"{request.symbol} from Yahoo Finance",
    ),
}


def get_available_sources() -> List[str]:
    """Return the registered source names, for CLI ``choices``.

    Returns:
        Source names in registration order.
    """
    return list(DOWNLOAD_SOURCES.keys())


def get_source(name: str) -> DownloadSource:
    """Look up a source by its registered name.

    Args:
        name: Registered source name, e.g. ``'ccxt'``.

    Returns:
        The registration, not a downloader instance.

    Raises:
        ValueError: If the name is not registered. The message lists what is.
    """
    if name not in DOWNLOAD_SOURCES:
        available = ', '.join(DOWNLOAD_SOURCES.keys())
        raise ValueError(f"Unknown data source: {name}. Available sources: {available}")
    return DOWNLOAD_SOURCES[name]


def get_source_option_names(name: str) -> Set[str]:
    """Return the argument names a source's ``download()`` accepts.

    Derived from the signature, so a new source's options need no second list.
    Callers use this to reject a foreign option with a message naming the source
    rather than letting it be ignored.

    Args:
        name: Registered source name.

    Returns:
        The accepted keyword argument names.

    Raises:
        ValueError: If the name is not registered.
    """
    source = get_source(name)
    signature = inspect.signature(source.downloader_class.download)
    return {
        parameter_name
        for parameter_name, parameter in signature.parameters.items()
        if parameter_name != 'self'
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


def create_downloader(name: str, options: Optional[Dict[str, Any]] = None) -> BaseDownloader:
    """Construct a registered source's downloader.

    Args:
        name: Registered source name.
        options: Keyword arguments for the downloader's ``__init__``.

    Returns:
        The constructed downloader.

    Raises:
        ValueError: If the name is not registered, or an option is not accepted by
            the downloader's constructor.
    """
    downloader_class = get_source(name).downloader_class
    kwargs = dict(options or {})

    signature = inspect.signature(downloader_class.__init__)
    accepted = {
        parameter_name for parameter_name in signature.parameters if parameter_name != 'self'
    }
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"Source '{name}' ({downloader_class.__name__}) does not accept: "
            f"{', '.join(unknown)}. It accepts: {', '.join(sorted(accepted))}"
        )

    return downloader_class(**kwargs)


def build_request(name: str, symbol: str, timeframe: str, start_date: str, end_date: str,
                  start: pd.Timestamp, end: pd.Timestamp,
                  options: Optional[Dict[str, Any]] = None) -> DownloadRequest:
    """Assemble a request for a registered source, applying its option defaults.

    Args:
        name: Registered source name.
        symbol: Trading pair or ticker, as typed.
        timeframe: Timeframe/interval, as typed.
        start_date: Raw start date string.
        end_date: Raw end date string.
        start: Parsed start timestamp.
        end: Parsed end timestamp.
        options: Source-specific options the user supplied.

    Returns:
        The request, with the source's option defaults filled in.

    Raises:
        ValueError: If the name is not registered, or an option is not accepted by
            the source.
    """
    source = get_source(name)
    accepted = get_source_option_names(name)

    supplied = dict(options or {})
    unknown = sorted(set(supplied) - accepted)
    if unknown:
        raise ValueError(
            f"Source '{name}' does not accept: {', '.join(unknown)}. "
            f"It accepts: {', '.join(sorted(accepted))}"
        )

    resolved = dict(source.option_defaults)
    resolved.update(supplied)

    return DownloadRequest(
        source=name, symbol=symbol, timeframe=timeframe,
        start_date=start_date, end_date=end_date, start=start, end=end,
        options=resolved,
    )


def build_download_kwargs(name: str, request: DownloadRequest) -> Dict[str, Any]:
    """Translate a request into the source's ``download()`` keyword arguments.

    The result is checked against the signature, so a registration that names an
    argument its downloader does not have fails here with a message rather than as
    an opaque ``TypeError`` at call time.

    Args:
        name: Registered source name.
        request: The request to translate.

    Returns:
        Keyword arguments for ``downloader.download``.

    Raises:
        ValueError: If the name is not registered, or the registration produced an
            argument the downloader does not accept.
    """
    source = get_source(name)
    kwargs = source.build_download_kwargs(request)

    accepted = get_source_option_names(name)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"Source '{name}' ({source.downloader_class.__name__}) does not accept: "
            f"{', '.join(unknown)}. It accepts: {', '.join(sorted(accepted))}"
        )

    return kwargs
