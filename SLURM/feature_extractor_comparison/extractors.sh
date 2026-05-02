#!/bin/bash

# Shared extractor order for the feature-extractor comparison workflow.
# Keep this list in sync with the config files and the array submitter.

EXTRACTORS=(
  "dynamics_original"
  "dynamics_no_conv"
  "mlp_small"
  "mlp_large"
  "attention_small"
  "lightweight_cnn"
)
