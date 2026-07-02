"""Pytest configuration: logging and shared fixtures."""
import logging


def pytest_configure(config: object) -> None:
    """Suppress library noise during test runs; set WARNING floor."""
    logging.basicConfig(level=logging.WARNING)
