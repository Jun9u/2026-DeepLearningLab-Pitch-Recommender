from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from web.backend.frontend import add_frontend_routes


class FrontendTests(unittest.TestCase):
    def test_registers_page_and_mounts_bundled_assets(self):
        with TemporaryDirectory() as directory:
            index = Path(directory) / "index.html"
            index.write_text("<!doctype html>", encoding="utf-8")
            assets = index.with_name("assets")
            assets.mkdir()

            app = FastAPI()
            add_frontend_routes(app, index)

            mount = next(route for route in app.routes if route.path == "/assets")
            self.assertEqual(mount.app.directory, assets)
            page = next(route for route in app.routes if route.path == "/").endpoint()
            self.assertIsInstance(page, FileResponse)
            self.assertEqual(page.path, index)

    def test_keeps_the_documentation_link_when_frontend_is_absent(self):
        with TemporaryDirectory() as directory:
            app = FastAPI()
            add_frontend_routes(app, Path(directory) / "missing" / "index.html")

            self.assertFalse(any(route.path == "/assets" for route in app.routes))
            page = next(route for route in app.routes if route.path == "/").endpoint()
            self.assertIsInstance(page, HTMLResponse)
            self.assertIn(b'href="/docs"', page.body)
