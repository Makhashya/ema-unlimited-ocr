"""Streamlit Community Cloud entrypoint: runs ../app.py.

Lives in its own folder so the deploy installs web/requirements.txt (the
light API/web set) instead of the GPU stack in the repo root -- Community
Cloud prefers dependency files beside the entrypoint. In the deploy dialog
set "Main file path" to web/streamlit_app.py.

Locally this is equivalent to `streamlit run app.py`.
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
