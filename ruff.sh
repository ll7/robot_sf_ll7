#!/usr/bin/env bash

# Fix ruff linting
uvx ruff check --fix

# Fix format
uvx ruff format
