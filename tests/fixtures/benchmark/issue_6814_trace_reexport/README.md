# Issue #6814 synthetic trace-re-export fixture

Tests build the compact package and three external arm roots in a temporary
directory so every fixture digest is computed from the bytes under test. The
fixture deliberately uses source-owned actor IDs and a producing-commit
configuration snapshot; no real benchmark output or release statistic is
stored here.
