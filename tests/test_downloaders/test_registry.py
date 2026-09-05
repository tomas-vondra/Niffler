"""
The shared data-source contract.

These tests iterate the registry, so a newly registered source is covered
automatically - the same arrangement as tests/test_strategies/test_registry.py.
They pin what lets scripts/download_data.py construct and invoke a source without
knowing its name: the choices come from the registry, the accepted options come
from the downloader's own download() signature, and every registration translates
a request into arguments that signature actually has.
"""

import inspect
import unittest
from dataclasses import replace
from unittest import mock

import pandas as pd

from niffler.data.downloaders.base_downloader import BaseDownloader
from niffler.data.downloaders.ccxt_downloader import CCXTDownloader
from niffler.data.downloaders.registry import (
    DOWNLOAD_SOURCES,
    DownloadSource,
    build_download_kwargs,
    build_request,
    create_downloader,
    get_available_sources,
    get_source,
    get_source_option_names,
)
from niffler.data.downloaders.yahoo_finance_downloader import YahooFinanceDownloader


class _ProbeDownloader(BaseDownloader):
    """A throwaway downloader with a distinctive argument, registered in one edit."""

    def __init__(self):
        super().__init__('Probe Downloader')

    def download(self, symbol: str, probe_venue: str = 'unset') -> pd.DataFrame:
        return pd.DataFrame({'symbol': [symbol], 'probe_venue': [probe_venue]})

    def get_supported_timeframes(self) -> list:
        return ['1d']


def request_for(source: str, **overrides):
    """Build a request with the fields download_data.py would supply."""
    values = {
        'symbol': 'BTC/USDT',
        'timeframe': '1d',
        'start_date': '2024-01-01',
        'end_date': '2024-01-05',
        'start': pd.Timestamp('2024-01-01'),
        'end': pd.Timestamp('2024-01-05'),
    }
    values.update(overrides)
    return build_request(source, **values)


class TestRegistryContract(unittest.TestCase):
    """Properties every registered source must have."""

    def test_every_registration_names_a_downloader(self):
        for name, source in DOWNLOAD_SOURCES.items():
            with self.subTest(source=name):
                self.assertTrue(issubclass(source.downloader_class, BaseDownloader))

    def test_every_downloader_constructs_with_no_options(self):
        """download_data.py builds a source without knowing its constructor."""
        for name in get_available_sources():
            with self.subTest(source=name):
                self.assertIsInstance(create_downloader(name), BaseDownloader)

    def test_every_registration_produces_arguments_download_accepts(self):
        """A registration that names a wrong argument fails here, not at call time."""
        for name in get_available_sources():
            with self.subTest(source=name):
                kwargs = build_download_kwargs(name, request_for(name))
                signature = inspect.signature(
                    get_source(name).downloader_class.download
                )
                signature.bind(None, **kwargs)

    def test_no_downloader_swallows_arguments_with_var_keyword(self):
        """**kwargs would make the derived option set 'everything'."""
        for name, source in DOWNLOAD_SOURCES.items():
            with self.subTest(source=name):
                kinds = [
                    parameter.kind
                    for parameter in inspect.signature(
                        source.downloader_class.download
                    ).parameters.values()
                ]
                self.assertNotIn(inspect.Parameter.VAR_KEYWORD, kinds)

    def test_unknown_source_lists_the_known_ones(self):
        with self.assertRaises(ValueError) as context:
            get_source('alpaca')

        message = str(context.exception)
        self.assertIn('Unknown data source: alpaca', message)
        for name in get_available_sources():
            self.assertIn(name, message)


class TestOptionsAreDerivedFromTheSignature(unittest.TestCase):
    """--exchange belongs to ccxt because CCXTDownloader.download says so."""

    def test_ccxt_accepts_an_exchange_id(self):
        self.assertIn('exchange_id', get_source_option_names('ccxt'))

    def test_yahoo_does_not_accept_an_exchange_id(self):
        self.assertNotIn('exchange_id', get_source_option_names('yahoo'))

    def test_a_foreign_option_raises_naming_the_accepted_ones(self):
        with self.assertRaises(ValueError) as context:
            request_for('yahoo', options={'exchange_id': 'binance'})

        message = str(context.exception)
        self.assertIn('does not accept: exchange_id', message)
        self.assertIn('ticker', message)


class TestTranslationIsByteIdentical(unittest.TestCase):
    """The two sources keep the exact call the CLI used to make by hand."""

    def test_ccxt_gets_millisecond_epochs_and_the_exchange(self):
        kwargs = build_download_kwargs('ccxt', request_for('ccxt'))

        self.assertEqual(kwargs, {
            'exchange_id': 'binance',
            'symbol': 'BTC/USDT',
            'timeframe': '1d',
            'start_ms': int(pd.Timestamp('2024-01-01').timestamp() * 1000),
            'end_ms': int(pd.Timestamp('2024-01-05').timestamp() * 1000),
        })

    def test_yahoo_gets_the_date_strings_exactly_as_typed(self):
        """Reformatting a date could change what the venue is asked for."""
        kwargs = build_download_kwargs(
            'yahoo', request_for('yahoo', symbol='SPY', start_date='2024-1-1')
        )

        self.assertEqual(kwargs, {
            'ticker': 'SPY',
            'start_date': '2024-1-1',
            'end_date': '2024-01-05',
            'interval': '1d',
        })

    def test_ccxt_defaults_to_binance_when_no_exchange_was_passed(self):
        self.assertEqual('binance', request_for('ccxt').options['exchange_id'])

    def test_an_explicit_exchange_wins(self):
        request = request_for('ccxt', options={'exchange_id': 'bybit'})

        self.assertEqual('bybit', request.options['exchange_id'])

    def test_the_file_tag_is_the_venue_for_ccxt_and_the_name_for_yahoo(self):
        """The default output filename used to need its own if/else."""
        self.assertEqual(
            'binance', get_source('ccxt').file_tag(request_for('ccxt'))
        )
        self.assertEqual(
            'yahoo', get_source('yahoo').file_tag(request_for('yahoo'))
        )

    def test_registered_classes_are_the_real_downloaders(self):
        self.assertIs(get_source('ccxt').downloader_class, CCXTDownloader)
        self.assertIs(get_source('yahoo').downloader_class, YahooFinanceDownloader)


class TestOneEditRegistration(unittest.TestCase):
    """A new source is one entry, and its argument arrives intact."""

    def setUp(self):
        probe = DownloadSource(
            downloader_class=_ProbeDownloader,
            build_download_kwargs=lambda request: {
                'symbol': request.symbol,
                'probe_venue': request.options['probe_venue'],
            },
            option_defaults={'probe_venue': 'probe-default'},
        )
        patcher = mock.patch.dict(DOWNLOAD_SOURCES, {'probe': probe})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_new_source_is_offered_to_the_cli(self):
        self.assertIn('probe', get_available_sources())

    def test_the_new_source_receives_its_own_option(self):
        kwargs = build_download_kwargs(
            'probe', request_for('probe', options={'probe_venue': 'niffler-probe'})
        )

        self.assertEqual('niffler-probe', kwargs['probe_venue'])
        frame = create_downloader('probe').download(**kwargs)
        self.assertEqual('niffler-probe', frame['probe_venue'].iloc[0])

    def test_an_option_the_new_source_lacks_raises(self):
        with self.assertRaises(ValueError) as context:
            request_for('probe', options={'exchange_id': 'binance'})

        message = str(context.exception)
        self.assertIn('does not accept: exchange_id', message)
        self.assertIn('probe_venue', message)

    def test_a_registration_naming_a_wrong_argument_is_caught(self):
        """The registration is data; a typo in it must not reach download().'"""
        broken = replace(
            DOWNLOAD_SOURCES['probe'],
            build_download_kwargs=lambda request: {'symbol': request.symbol,
                                                   'probe_venu': 'typo'},
        )
        with mock.patch.dict(DOWNLOAD_SOURCES, {'probe': broken}):
            with self.assertRaises(ValueError) as context:
                build_download_kwargs('probe', request_for('probe'))

        self.assertIn('probe_venu', str(context.exception))


if __name__ == '__main__':
    unittest.main()
