# Test package for script main functions

import os

# Every CLI reads ./niffler.toml by default (scripts.config_file). These tests
# drive main() with a patched sys.argv and assert on the result, so a
# developer's own config file must not reach them. An empty value switches file
# loading off; the config tests pass --config explicitly, which still wins.
os.environ['NIFFLER_CONFIG'] = ''
