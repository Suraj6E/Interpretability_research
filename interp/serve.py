"""Live mode: a local server that runs the model for the viewer.

    python -m interp.serve                  # GPT-2 small at http://127.0.0.1:8000
    python -m interp.serve --model gpt2-small --port 8001

It listens only on this computer (127.0.0.1) unless --host says otherwise.
If the viewer has been built (`npm run build` in viewer/), it is served at /.
"""

import argparse
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from interp.model import load_model, model_spec
from interp.results import analyze_prompt

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIST = ROOT / "viewer" / "dist"
RESULTS_DIR = ROOT / "results"
MAX_PROMPT_CHARS = 2000


class AnalyzeRequest(BaseModel):
    prompt: str
    correct: str | None = None  # answer tokens DLA compares; default: the top two guesses
    wrong: str | None = None


def create_app(model, spec) -> FastAPI:
    app = FastAPI(title="interp live server")
    lock = threading.Lock()  # one model in memory, so one request at a time

    @app.get("/api/info")
    def info():
        return {
            "model": spec.key,
            "revision": spec.revision,
            "device": str(model.cfg.device),
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
        }

    @app.post("/api/analyze")
    def analyze(request: AnalyzeRequest):
        if not request.prompt.strip():
            raise HTTPException(400, "The prompt is empty.")
        if len(request.prompt) > MAX_PROMPT_CHARS:
            raise HTTPException(400, f"The prompt is longer than {MAX_PROMPT_CHARS} characters.")
        with lock:
            try:
                return analyze_prompt(
                    model, request.prompt, spec.key, request.correct, request.wrong
                )
            except ValueError as error:  # too long, or an answer that isn't one token
                raise HTTPException(400, str(error)) from error

    # Saved batch results, for the viewer's question picker.
    if RESULTS_DIR.exists():
        app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
    # Registered last, so /api/* and /results/* are never shadowed by the viewer files.
    if VIEWER_DIST.exists():
        app.mount("/", StaticFiles(directory=VIEWER_DIST, html=True), name="viewer")
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt2-small", help="key in configs/models.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    spec = model_spec(args.model)
    model = load_model(spec)
    if VIEWER_DIST.exists():
        print(f"Open http://{args.host}:{args.port} in your browser.")
    else:
        print("Viewer not built: run `npm run build` in viewer/, or `npm run dev` there for development.")
    uvicorn.run(create_app(model, spec), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
