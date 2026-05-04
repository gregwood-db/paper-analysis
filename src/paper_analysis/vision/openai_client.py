from __future__ import annotations

import base64
import json
import os

from openai import OpenAI
from pydantic import ValidationError

from paper_analysis.prompts import JSON_FIX_SUFFIX
from paper_analysis.vision.base import PlotType, _prompts_for, classify_prompts, parse_extraction_text

_VALID_PLOT_TYPES = frozenset({
    "box_plot", "line_chart", "line_plot", "heatmap", "table_image",
    "plasmid_map", "workflow_diagram", "experimental_workflow",
})


class OpenAIVisionClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._client = OpenAI()

    def _classify_figure(self, data_url: str) -> str:
        """Ask the model to classify the figure type before extraction."""
        system, user_text = classify_prompts()
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=64,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        raw = (resp.choices[0].message.content or "").strip().lower().replace(" ", "_")
        if raw in _VALID_PLOT_TYPES:
            return raw
        for valid in _VALID_PLOT_TYPES:
            if valid in raw:
                return valid
        return "table_image"

    def extract_figure(
        self,
        image_png: bytes,
        plot_type: PlotType,
        *,
        max_retries: int = 1,
    ):
        b64 = base64.standard_b64encode(image_png).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        if plot_type == "auto":
            plot_type = self._classify_figure(data_url)

        system, user_text = _prompts_for(plot_type)
        attempts = max_retries + 1
        last_err: Exception | None = None
        for attempt in range(attempts):
            text = user_text if attempt == 0 else user_text + JSON_FIX_SUFFIX
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=16384,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            choice = resp.choices[0]
            last_text = choice.message.content or ""
            try:
                return parse_extraction_text(last_text)
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                last_err = e
                if attempt >= attempts - 1:
                    raise
        raise RuntimeError(f"extraction failed: {last_err}")
