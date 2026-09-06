"""Shared visual theme - the background gradient injected on every dashboard
page (``app.py`` and ``pages/*.py``), so the app reads as one consistent
design instead of two pages with different looks bolted together.

Page-specific styling (the stat cards' glass treatment, the hazard banner,
the sidebar legend) stays local to the page that actually uses it - this
module only holds what every page shares.
"""

from __future__ import annotations

BACKGROUND_CSS = """
<style>
[data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(180deg, #dff1fb 0%, #eef3f4 45%, #f5f5f3 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: transparent;
}
</style>
"""
