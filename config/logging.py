"""Backwards-compatible shim for the old ``config.logging`` import path.

The real implementation now lives in :mod:`niffler.config.logging`; this module
only re-exports it so existing imports (``from config.logging import
setup_logging``) keep working.  New code should import from
``niffler.config.logging`` directly.
"""

from niffler.config.logging import (
    DEFAULT_LOG_FORMAT,
    VALID_LOG_LEVELS,
    resolve_log_level,
    setup_logging,
)

__all__ = [
    'DEFAULT_LOG_FORMAT',
    'VALID_LOG_LEVELS',
    'resolve_log_level',
    'setup_logging'
]
