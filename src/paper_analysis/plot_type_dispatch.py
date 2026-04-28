from __future__ import annotations

import warnings
from typing import Any

from paper_analysis.schemas import (
    BoxPlotExtraction,
    ExperimentalWorkflowExtraction,
    LineChartExtraction,
    PlasmidMapExtraction,
    TableImageExtraction,
    UnknownPlotExtraction,
    WorkflowDiagramExtraction,
)

ExtractionModel = (
    BoxPlotExtraction
    | LineChartExtraction
    | TableImageExtraction
    | PlasmidMapExtraction
    | WorkflowDiagramExtraction
    | ExperimentalWorkflowExtraction
    | UnknownPlotExtraction
)


def warn_unknown_plot_type(declared: object, *, context: str) -> None:
    label = repr(declared) if declared is not None else "missing"
    warnings.warn(
        f"Unknown plot_type {label} ({context}). "
        "Add a schema in paper_analysis/schemas.py, prompts in prompts.py, "
        "and register the type in plot_type_dispatch.parse_extraction_dict.",
        UserWarning,
        stacklevel=3,
    )


def _coerce_unknown(raw: dict[str, Any], declared: object) -> UnknownPlotExtraction:
    dt = declared if isinstance(declared, str) else None
    return UnknownPlotExtraction(declared_plot_type=dt, raw=dict(raw))


_PLOT_TYPE_ALIASES: dict[str, str] = {
    "line_plot": "line_chart",
    "circular_genome_map": "plasmid_map",
    "genome_map": "plasmid_map",
    "vector_map": "plasmid_map",
    "schematic_diagram": "workflow_diagram",
    "schematic": "workflow_diagram",
    "flow_diagram": "workflow_diagram",
    "flowchart": "workflow_diagram",
    "protocol_diagram": "experimental_workflow",
    "bar_chart": "box_plot",
    "bar_plot": "box_plot",
}


def parse_extraction_dict(data: dict[str, Any], *, context: str) -> ExtractionModel:
    """Route parsed JSON to the correct Pydantic model; unknown types become UnknownPlotExtraction with a warning."""
    pt = data.get("plot_type")
    canonical = _PLOT_TYPE_ALIASES.get(pt, pt) if isinstance(pt, str) else pt
    if canonical != pt and isinstance(pt, str):
        data = {**data, "plot_type": canonical}
    if canonical == "unknown":
        return UnknownPlotExtraction.model_validate(data)
    if canonical == "box_plot":
        return BoxPlotExtraction.model_validate(data)
    if canonical == "line_chart":
        return LineChartExtraction.model_validate(data)
    if canonical == "table_image":
        return TableImageExtraction.model_validate(data)
    if canonical == "plasmid_map":
        return PlasmidMapExtraction.model_validate(data)
    if canonical == "workflow_diagram":
        return WorkflowDiagramExtraction.model_validate(data)
    if canonical == "experimental_workflow":
        return ExperimentalWorkflowExtraction.model_validate(data)
    warn_unknown_plot_type(pt, context=context)
    return _coerce_unknown(data, pt)
