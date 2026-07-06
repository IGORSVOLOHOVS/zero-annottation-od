"""VLM backends used to auto-label images.

The primary backend shells out to llama.cpp's prebuilt ``llama-server``
binary and talks to its OpenAI-compatible ``/v1/chat/completions`` endpoint.
We deliberately do not use the ``llama-cpp-python`` bindings: on Windows its
CUDA wheels are either unofficial community builds or require a local
MSVC + CUDA toolchain to compile, whereas llama.cpp's own GitHub releases
ship a ready-to-run ``llama-server.exe`` (CPU or CUDA) with no build step.
See README.md for the full reasoning.
"""

from __future__ import annotations

import abc
import base64
import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import requests

from zeod.config import LlamaCppConfig
from zeod.labeling.grid import rescale_bbox_to_pixels, smart_resize

logger = logging.getLogger(__name__)

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".bmp": "image/bmp"}


class VLMBackend(abc.ABC):
    """Minimal interface a labeling backend must implement."""

    def __enter__(self) -> VLMBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:  # noqa: B027 - intentionally optional, MockBackend has nothing to close
        pass

    def restart(self) -> None:  # noqa: B027 - intentionally optional, default is a no-op
        """Recover from a wedged backend (e.g. a hung server process). Default: no-op."""
        pass

    def bbox_rescaler(self) -> Callable[[list[float], int, int], list[float]] | None:
        """Optional (bbox, orig_w, orig_h) -> bbox hook for backends whose model
        emits coordinates in a resized/internal grid rather than original pixels."""
        return None

    @abc.abstractmethod
    def generate(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return the raw text response for a single image."""
        raise NotImplementedError


def _image_to_data_url(image_path: Path) -> str:
    mime = _MIME_TYPES.get(image_path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class LlamaCppServerBackend(VLMBackend):
    """Manages a `llama-server` subprocess and queries it over HTTP."""

    def __init__(self, config: LlamaCppConfig):
        self.config = config
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        cfg = self.config
        cmd = [
            cfg.server_binary,
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
            "--ctx-size",
            str(cfg.ctx_size),
            "--n-gpu-layers",
            str(cfg.n_gpu_layers),
            "--image-min-tokens",
            str(cfg.image_min_tokens),
            "--parallel",
            str(cfg.n_parallel),
        ]
        if cfg.model_path:
            cmd += ["--model", str(cfg.model_path)]
            if cfg.mmproj_path:
                cmd += ["--mmproj", str(cfg.mmproj_path)]
        elif cfg.hf_repo:
            cmd += ["-hf", f"{cfg.hf_repo}:{cfg.hf_quant}"]
        else:
            raise ValueError("llama_cpp config needs either hf_repo or model_path")
        cmd += list(cfg.extra_server_args)

        logger.info("Starting llama-server: %s", " ".join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Could not launch '{cfg.server_binary}'. Download a llama.cpp release for "
                "Windows from https://github.com/ggml-org/llama.cpp/releases, unzip it, and "
                "either add its folder to PATH or set llama_cpp.server_binary to the full "
                "path of llama-server.exe."
            ) from e

        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        health_url = f"{self.config.base_url}/health"
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                tail = self._process.stdout.read() if self._process.stdout else ""
                raise RuntimeError(f"llama-server exited early (code {self._process.returncode}):\n{tail}")
            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    logger.info("llama-server is ready at %s", self.config.base_url)
                    return
            except requests.RequestException as e:
                last_error = e
            time.sleep(2)
        raise TimeoutError(
            f"llama-server did not become ready within {self.config.startup_timeout_s}s (last error: {last_error})"
        )

    def generate(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.config.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.config.request_timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def close(self) -> None:
        if self._process is None:
            return
        logger.info("Stopping llama-server (pid %s)", self._process.pid)
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def restart(self) -> None:
        logger.warning("Restarting llama-server after repeated failures")
        self.close()
        self.start()

    def bbox_rescaler(self) -> Callable[[list[float], int, int], list[float]]:
        min_pixels, max_pixels = self.config.min_pixels, self.config.max_pixels

        def _rescale(bbox: list[float], orig_w: int, orig_h: int) -> list[float]:
            grid_w, grid_h = smart_resize(orig_w, orig_h, min_pixels, max_pixels)
            return rescale_bbox_to_pixels(bbox, orig_w, orig_h, grid_w, grid_h)

        return _rescale


class MockBackend(VLMBackend):
    """Deterministic backend for tests: returns a fixed or callable response.

    ``responder`` receives the image path and must return the raw text the
    "model" would have produced, so tests can simulate malformed JSON,
    empty detections, etc. without any real inference.
    """

    def __init__(self, responder: Callable[[Path], str] | str = "[]"):
        self._responder = responder

    def generate(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if callable(self._responder):
            return self._responder(image_path)
        return self._responder
