"""Streamlit-free pure-Python components for ``app/photon_app.py``.

Modules under this package MUST NOT import :mod:`streamlit`. Keeping them
framework-free lets the unit tests in ``tests/test_photon_app_components.py``
exercise the helpers without spinning up a Streamlit runtime.

The invariant is pinned by ``T-C-streamlit-absent`` which imports every
module in this package and asserts that ``"streamlit" not in sys.modules``
afterwards (via a fresh subprocess so the test runner's own imports do
not pollute the check).
"""
