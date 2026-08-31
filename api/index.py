"""Vercel serverless entry point.

Vercel's Python runtime discovers `app` in this module and serves it as an
ASGI application. Only /api/* is routed here. The static pages in public/
are served from the CDN, so a cold start (which loads TensorFlow and a
~49 MB checkpoint) is paid only on the first analysis, never on a page view.
"""

import sys
from pathlib import Path

# The bundler decides what to ship by reading imports statically, and cannot
# see a path inserted at runtime, so webapp/ is named in vercel.json's
# includeFiles as well, or the function ships without it and dies on import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp.server import app  # noqa: E402

__all__ = ["app"]
