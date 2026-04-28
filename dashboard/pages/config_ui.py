"""
Config UI page — read-only view of current runtime configuration.

Does NOT allow modifying any settings. Configuration changes must be made
via config/config.yaml and a bot restart.
"""

from __future__ import annotations

import os

import streamlit as st
import yaml


def render():
    st.header("Configuration (Read-Only)")
    st.caption("To change settings, edit config/config.yaml and restart the bot.")

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "config.yaml"
    )
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        st.json(config)
    except FileNotFoundError:
        st.warning(f"config.yaml not found at {config_path}")
    except Exception as e:
        st.error(f"Error loading config: {e}")


if __name__ == "__main__":
    render()
