"""Camera-ready campaign package.

Decomposes the historically monolithic ``camera_ready_campaign.py`` (~4967 LOC)
into focused submodules (see #3385). Leaf helpers live in ``_util.py``;
config dataclasses live in ``camera_ready_campaign_config.py``; the remaining
orchestration/IO/computation stays in ``camera_ready_campaign.py`` until later
slices extract it.
"""
