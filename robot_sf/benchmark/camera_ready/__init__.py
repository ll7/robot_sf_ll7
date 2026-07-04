"""Camera-ready campaign package.

Decomposes the historically monolithic ``camera_ready_campaign.py`` (~4967 LOC)
into focused submodules (see #3385). Leaf helpers live in ``_util.py``;
config dataclasses live in ``_config_types.py`` while
``camera_ready_campaign_config.py`` remains a compatibility facade. Most
orchestration/IO/computation has moved into this package while
``camera_ready_campaign.py`` preserves legacy import and monkeypatch surfaces.
"""
