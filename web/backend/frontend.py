"""Serve the browser page and its bundled static assets."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


def add_frontend_routes(app: FastAPI, index_path: Path):
    assets_path = index_path.with_name("assets")
    assets_mounted = assets_path.is_dir()
    if assets_mounted:
        app.mount("/assets", StaticFiles(directory=assets_path), name="frontend-assets")

    required_files = (index_path, assets_path / "styles.css", assets_path / "app.js")

    @app.get("/")
    def index():
        if assets_mounted and all(path.is_file() for path in required_files):
            return FileResponse(index_path)

        return HTMLResponse(
            """
        <h1>Pitch Recommendation API</h1>
        <p>Frontend file not found. Use <a href="/docs">/docs</a> to test API.</p>
        """
        )
