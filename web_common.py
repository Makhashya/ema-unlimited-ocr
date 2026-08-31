"""Bits shared by the web UIs (app.py, app_jobs.py).

  load_streamlit_secrets  st.secrets -> environment (hosted deployments)
  require_password        APP_PASSWORD sign-in gate
  cli_available           is the `claude` CLI installed here?
"""

import hmac
import os

import streamlit as st

import equipment_pipeline as cli_pipe


def load_streamlit_secrets() -> None:
    """Copy string entries of st.secrets into the environment.

    Streamlit Community Cloud (and a local .streamlit/secrets.toml) expose
    settings through st.secrets; the pipelines read .env-style variables
    (OPENAI_API_KEY, VLM_PROVIDER, ...). Real environment variables win.
    """
    try:
        items = list(st.secrets.items())
    except Exception:                       # no secrets file: local use
        return
    for key, value in items:
        if isinstance(value, str) and value:
            os.environ.setdefault(key, value)


def require_password(title: str = "🔧 Equipment List Extractor") -> None:
    """Gate the page behind APP_PASSWORD when it is set.

    Without APP_PASSWORD (environment, .env, or secrets) the app is open,
    which is fine on a private machine; set it before exposing the app.
    """
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("authed"):
        return
    st.title(title)
    with st.form("login"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            if hmac.compare_digest(password, expected):
                st.session_state["authed"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()


def cli_available() -> bool:
    """True when the `claude` CLI can be found on this machine."""
    try:
        cli_pipe.find_claude(None)
        return True
    except SystemExit:
        return False
