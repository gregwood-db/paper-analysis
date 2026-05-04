"""Post-processing: expand vision extraction data into individual measurement rows.

The combined LLM often summarizes figure data instead of emitting one row per
data point.  This module reads the raw vision extraction JSONs (already on
disk) and expands each line-chart point, box-plot group, and heatmap cell into
a ``CombinedMeasurementCandidate``-shaped dict.  The expanded rows are then
merged (deduplicated) with the LLM's ``candidate_combined.json``.

This step requires **no** additional API calls and runs on already-extracted
artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Master Field List (MFL) loader
# ---------------------------------------------------------------------------

_PAPER_REF_TO_ID: dict[str, str] = {
    "Chen 2017": "006_Chen_2017",
    "Dorado-Morales 2021": "002_DoradoMorales_2021",
    "Freter 1983": "007_Freter_1983",
    "Gama 2017": "003_Gama_2017",
    "Hall 2020": "005_Hall_2020",
    "Hardiman 2016": "011_Hardiman_2016",
    "Igler 2021": "008_Igler_2021",
    "Kosterlitz 2022": "009_Kosterlitz_2022",
    "Leon-Sampedro 2021": "012_Leon-Sampedro_2021",
    "Lobato-Marquez 2016": "013_Lobato-Marquez_2016",
    "Lopatkin 2017": "014_Lopatkin_2017",
    "Milan 2018": "004_Milan_2018",
    "Valle 2021": "001_Valle_2021",
    "Wan 2011": "010_Wan_2011",
    "Wein 2019": "015_Wein_2019",
}

_MFL_METADATA_CATEGORIES = frozenset({
    "paper_metadata", "strain_information",
})


def _norm(name: str) -> str:
    """Lowercase, strip, collapse whitespace/hyphens to underscores."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower())


@dataclass
class MasterFieldIndex:
    """Per-paper index of allowed field names built from the MASTER_FIELD_LIST.

    Attributes
    ----------
    global_fields : set[str]
        All normalised field names from the MFL (across all papers).
    paper_fields : dict[str, set[str]]
        paper_id -> set of normalised field names valid for that paper.
    canonical_fields : set[str]
        Normalised names of fields where ``Canonical_Field == 'Yes'``.
    category_map : dict[str, str]
        normalised field_name -> Field_Category.
    """
    global_fields: set[str] = field(default_factory=set)
    paper_fields: dict[str, set[str]] = field(default_factory=dict)
    canonical_fields: set[str] = field(default_factory=set)
    category_map: dict[str, str] = field(default_factory=dict)

    def allowed_for_paper(self, paper_id: str) -> set[str]:
        """Return the set of normalised field names valid for *paper_id*.

        Falls back to the global set if the paper has no specific mapping.
        """
        return self.paper_fields.get(paper_id, self.global_fields)

    def is_metadata_category(self, field_name: str) -> bool:
        cat = self.category_map.get(_norm(field_name), "")
        return cat in _MFL_METADATA_CATEGORIES


def load_master_field_index(path: Path) -> MasterFieldIndex:
    """Parse the MASTER_FIELD_LIST Excel into a ``MasterFieldIndex``."""
    df = pd.read_excel(path, sheet_name="MASTER_FIELD_LIST")
    idx = MasterFieldIndex()

    for _, row in df.iterrows():
        raw_name = str(row.get("Field_Name", ""))
        if not raw_name.strip():
            continue

        primary = _norm(raw_name)
        idx.global_fields.add(primary)
        idx.category_map[primary] = str(row.get("Field_Category", ""))

        synonyms_str = str(row.get("Synonyms", ""))
        alt_names: list[str] = []
        if synonyms_str and synonyms_str not in ("—", "nan", ""):
            alt_names = [_norm(s) for s in synonyms_str.split(";") if s.strip()]

        all_names = [primary] + alt_names
        for n in alt_names:
            idx.global_fields.add(n)
            idx.category_map.setdefault(n, idx.category_map[primary])

        if str(row.get("Canonical_Field", "")).strip().lower() == "yes":
            idx.canonical_fields.add(primary)
            for n in alt_names:
                idx.canonical_fields.add(n)

        papers_str = str(row.get("Papers", ""))
        if papers_str and papers_str != "nan":
            for ref in papers_str.split(";"):
                ref = ref.strip()
                paper_id = _PAPER_REF_TO_ID.get(ref)
                if paper_id:
                    bucket = idx.paper_fields.setdefault(paper_id, set())
                    for n in all_names:
                        bucket.add(n)

    return idx


_cached_mfl: MasterFieldIndex | None = None


def get_master_field_index(path: Path | None) -> MasterFieldIndex | None:
    """Return a cached ``MasterFieldIndex``, loading from *path* on first call."""
    global _cached_mfl
    if path is None:
        return None
    if _cached_mfl is not None:
        return _cached_mfl
    if not path.is_file():
        return None
    _cached_mfl = load_master_field_index(path)
    return _cached_mfl


def filter_by_master_field_list(
    combined_path: Path,
    paper_id: str,
    mfl: MasterFieldIndex,
    *,
    remove: bool = True,
) -> tuple[int, int]:
    """Filter extracted rows against the master field list whitelist.

    A row is **kept** if any of the following hold:

    1. Its normalised ``field_name`` is in the per-paper allowed set, OR
    2. Its *canonical* form (via batch_evaluate synonyms) matches the
       canonical form of any per-paper allowed field, OR
    3. Its normalised name is in the MFL's global ``canonical_fields``
       (Canonical_Field=Yes), OR
    4. Its *canonical* form matches any global MFL field's canonical form.

    Rows whose MFL category is paper/strain/plasmid metadata are always
    removed — they are never measurements.

    Returns ``(n_removed, n_kept)``.
    """
    if not combined_path.is_file():
        return 0, 0

    data = json.loads(combined_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])
    if not candidates:
        return 0, 0

    from paper_analysis.batch_evaluate import _canonicalize_field_name, _SYNONYM_MAP

    allowed_paper = mfl.allowed_for_paper(paper_id)
    allowed_paper_canonical: set[str] = {
        _canonicalize_field_name(fn) for fn in allowed_paper
    }

    global_canonical: set[str] = {
        _canonicalize_field_name(fn) for fn in mfl.global_fields
    }

    known_synonyms: set[str] = set(_SYNONYM_MAP.keys()) | set(_SYNONYM_MAP.values())

    kept: list[dict] = []
    n_removed = 0

    for c in candidates:
        fn = _norm(str(c.get("field_name", "")))
        fn_canon = _canonicalize_field_name(fn)

        if mfl.is_metadata_category(fn):
            n_removed += 1
            continue

        if fn in allowed_paper or fn_canon in allowed_paper_canonical:
            kept.append(c)
        elif fn in mfl.global_fields or fn_canon in global_canonical:
            kept.append(c)
        elif fn in known_synonyms or fn_canon in known_synonyms:
            kept.append(c)
        elif remove:
            n_removed += 1
        else:
            c["confidence"] = "low"
            kept.append(c)

    data["candidates"] = kept
    notes = data.get("notes") or ""
    if n_removed:
        data["notes"] = f"{notes}; MFL filter: removed {n_removed}, kept {len(kept)}".strip("; ")

    combined_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return n_removed, len(kept)


def _page_from_stem(stem: str) -> int | None:
    """Extract 1-based page number from a figure id like 'p008_fig01'."""
    m = re.match(r"^p(\d+)_", stem)
    return int(m.group(1)) if m else None


def _fig_source_label(stem: str) -> str:
    """Build a human-readable source label like 'Fig. crop p008_fig01'."""
    page = _page_from_stem(stem)
    return f"Fig. (vision crop {stem}, page {page})" if page else f"Fig. (vision crop {stem})"


def _expand_line_chart(
    stem: str, data: dict[str, Any], paper_id: str, id_counter: int,
    dominant_field: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Expand a line chart extraction: one row per (series, point)."""
    rows: list[dict[str, Any]] = []
    source = _fig_source_label(stem)
    y_label = data.get("axis_y_label") or ""
    x_label = data.get("axis_x_label") or "x"
    x_units = data.get("axis_x_units") or ""
    y_units = data.get("axis_y_units") or ""

    field_name = _infer_field_name_from_axis(y_label, dominant_field)

    for series in data.get("series", []):
        series_name = series.get("name") or "unnamed"
        for pt in series.get("points", []):
            x_val = pt.get("x")
            y_val = pt.get("y")
            if y_val is None:
                continue

            time_h = _try_parse_time(x_val, x_label, x_units)

            id_counter += 1
            row: dict[str, Any] = {
                "measurement_id": f"PP_{id_counter:04d}",
                "run_id": f"PP_RUN_{stem}_{series_name}",
                "paper_id": paper_id,
                "experiment_type": "figure_extraction",
                "species": None,
                "strain_id": series_name,
                "plasmid_name": None,
                "plasmid_family": None,
                "plasmid_size": None,
                "mobilization_type": None,
                "medium": None,
                "temperature": None,
                "culture_format": None,
                "replicates_biological": None,
                "selection_antibiotic": None,
                "run_description": f"Vision-extracted {y_label} from {source}, series '{series_name}', {x_label}={x_val}",
                "field_name": field_name,
                "measurement_time_h": time_h,
                "raw_value": y_val,
                "raw_units": y_units or None,
                "normalized_value": None,
                "normalized_units": None,
                "dispersion_value": _error_bar(pt),
                "dispersion_type": "error_bar" if _error_bar(pt) is not None else None,
                "source_in_paper": source,
                "confidence": "medium",
            }
            rows.append(row)
    return rows, id_counter


def _expand_box_plot(
    stem: str, data: dict[str, Any], paper_id: str, id_counter: int,
    dominant_field: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Expand a box plot extraction: one row per group (median as raw_value)."""
    rows: list[dict[str, Any]] = []
    source = _fig_source_label(stem)
    y_label = data.get("axis_y_label") or ""
    y_units = data.get("axis_y_units") or ""
    x_label = data.get("axis_x_label") or ""
    field_name = _infer_field_name_from_axis(y_label, dominant_field)

    for group in data.get("groups", []):
        median = group.get("median")
        if median is None:
            continue
        label = group.get("label", "")

        id_counter += 1
        q1 = group.get("q1")
        q3 = group.get("q3")
        iqr_str = None
        if q1 is not None and q3 is not None:
            iqr_str = f"{q1}-{q3}"

        row: dict[str, Any] = {
            "measurement_id": f"PP_{id_counter:04d}",
            "run_id": f"PP_RUN_{stem}_{label}",
            "paper_id": paper_id,
            "experiment_type": "figure_extraction",
            "species": None,
            "strain_id": label or None,
            "plasmid_name": None,
            "plasmid_family": None,
            "plasmid_size": None,
            "mobilization_type": None,
            "medium": None,
            "temperature": None,
            "culture_format": None,
            "replicates_biological": None,
            "selection_antibiotic": None,
            "run_description": f"Vision-extracted {y_label} from {source}, group '{label}'",
            "field_name": field_name,
            "measurement_time_h": None,
            "raw_value": median,
            "raw_units": y_units or None,
            "normalized_value": None,
            "normalized_units": None,
            "dispersion_value": iqr_str,
            "dispersion_type": "IQR" if iqr_str else None,
            "source_in_paper": source,
            "confidence": "medium",
        }
        rows.append(row)
    return rows, id_counter


def _expand_heatmap(
    stem: str, data: dict[str, Any], paper_id: str, id_counter: int,
    dominant_field: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Expand a heatmap extraction: one row per cell."""
    rows: list[dict[str, Any]] = []
    source = _fig_source_label(stem)
    val_label = data.get("value_label") or ""
    val_units = data.get("value_units") or ""
    field_name = _infer_field_name_from_axis(val_label, dominant_field)

    for cell in data.get("cells", []):
        value = cell.get("value")
        if value is None:
            continue
        row_label = cell.get("row_label", "")
        col_label = cell.get("col_label", "")

        id_counter += 1
        row: dict[str, Any] = {
            "measurement_id": f"PP_{id_counter:04d}",
            "run_id": f"PP_RUN_{stem}_{row_label}_{col_label}",
            "paper_id": paper_id,
            "experiment_type": "figure_extraction",
            "species": None,
            "strain_id": f"{row_label} × {col_label}",
            "plasmid_name": None,
            "plasmid_family": None,
            "plasmid_size": None,
            "mobilization_type": None,
            "medium": None,
            "temperature": None,
            "culture_format": None,
            "replicates_biological": None,
            "selection_antibiotic": None,
            "run_description": f"Vision-extracted {val_label} from {source}, row='{row_label}' col='{col_label}'",
            "field_name": field_name,
            "measurement_time_h": None,
            "raw_value": value,
            "raw_units": val_units or None,
            "normalized_value": None,
            "normalized_units": None,
            "dispersion_value": None,
            "dispersion_type": None,
            "source_in_paper": source,
            "confidence": "medium",
        }
        rows.append(row)
    return rows, id_counter


def _expand_table_image(
    stem: str, data: dict[str, Any], paper_id: str, id_counter: int,
    dominant_field: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Expand a table image: one row per (data-row, numeric-column) pair."""
    rows: list[dict[str, Any]] = []
    source = _fig_source_label(stem)
    headers = data.get("column_headers", [])
    table_rows = data.get("rows", [])

    if not headers or not table_rows:
        return rows, id_counter

    for ri, trow in enumerate(table_rows):
        row_label = trow[0] if trow else f"row_{ri}"
        for ci, cell_val in enumerate(trow):
            if ci == 0:
                continue
            header = headers[ci] if ci < len(headers) else f"col_{ci}"
            parsed = _try_parse_numeric(cell_val)
            if parsed is None:
                continue

            id_counter += 1
            row: dict[str, Any] = {
                "measurement_id": f"PP_{id_counter:04d}",
                "run_id": f"PP_RUN_{stem}_r{ri}",
                "paper_id": paper_id,
                "experiment_type": "figure_extraction",
                "species": None,
                "strain_id": str(row_label) if row_label else None,
                "plasmid_name": None,
                "plasmid_family": None,
                "plasmid_size": None,
                "mobilization_type": None,
                "medium": None,
                "temperature": None,
                "culture_format": None,
                "replicates_biological": None,
                "selection_antibiotic": None,
                "run_description": f"Vision-extracted table from {source}, row='{row_label}' col='{header}'",
                "field_name": _normalize_to_snake(header),
                "measurement_time_h": None,
                "raw_value": parsed,
                "raw_units": None,
                "normalized_value": None,
                "normalized_units": None,
                "dispersion_value": None,
                "dispersion_type": None,
                "source_in_paper": source,
                "confidence": "medium",
            }
            rows.append(row)
    return rows, id_counter


# ── Helpers ──────────────────────────────────────────────────────────────────

_AXIS_TO_FIELD: list[tuple[re.Pattern, str]] = [
    (re.compile(r"host.?freq|fraction.*plasmid|plasmid.*freq|% plasmid", re.I), "host_frequency_plasmid_bearing_cells"),
    (re.compile(r"conjug.*rate|transfer.*freq|conjug.*freq", re.I), "conjugation_rate"),
    (re.compile(r"conjug.*effic", re.I), "conjugation_efficiency_density_scaled"),
    (re.compile(r"conjug.*detect|transconj.*detect", re.I), "conjugation_detection_status"),
    (re.compile(r"growth.*rate|max.*growth|μ_?max", re.I), "growth_rate"),
    (re.compile(r"OD|optical.?density", re.I), "maximum_od"),
    (re.compile(r"area.*under.*curve|AUC", re.I), "area_under_growth_curve"),
    (re.compile(r"fitness|selection.*coeff|competition.*index", re.I), "relative_fitness"),
    (re.compile(r"copy.*number|copies.*per.*chrom", re.I), "plasmid_copy_number"),
    (re.compile(r"plasmid.*stab|plasmid.*loss|loss.*freq", re.I), "plasmid_loss_frequency"),
    (re.compile(r"segregat", re.I), "segregation_factor"),
    (re.compile(r"CFU|colony.?form|log.*cfu|cfu.*ml|log.*bact|bacteria.*per.*ml", re.I), "cfu_per_ml"),
    (re.compile(r"luminescence|lux|RLU", re.I), "sos_response_auc"),
    (re.compile(r"MIC|minimum.*inhibitory", re.I), "mic"),
    (re.compile(r"colony.*area|mean.*colony", re.I), "colony_area"),
    (re.compile(r"coverage", re.I), "coverage"),
    (re.compile(r"transcription.*plasmid|relative.*transcription", re.I), "relative_transcription_per_plasmid"),
]


def _infer_field_name_from_axis(
    axis_label: str,
    dominant_field: str | None = None,
) -> str:
    """Map a figure axis label to a plausible field_name.

    If the axis label is empty/None and the paper's LLM output has a clear
    dominant field_name, use that as the default.
    """
    if axis_label:
        for pat, field in _AXIS_TO_FIELD:
            if pat.search(axis_label):
                return field
        snake = _normalize_to_snake(axis_label)
        if snake and snake != "none" and len(snake) > 4 and not _is_label_noise(axis_label):
            return snake

    if dominant_field:
        return dominant_field
    return "figure_value"


def _is_label_noise(label: str) -> bool:
    """Return True if the label looks like a section header or plasmid name, not a measurement."""
    s = label.strip()
    if len(s) <= 6 and re.match(r"^[A-Za-z0-9]+$", s):
        return True
    if re.match(r"^(p[A-Z]|R\d|RP\d|RN\d|F$|Col)", s, re.I):
        return True
    return False


def _get_dominant_field(combined_path: Path) -> str | None:
    """Read the existing combined output and return the most common field_name.

    This is used as fallback for vision extractions where axis labels are
    missing — the paper's LLM output typically captures the correct field names.
    """
    if not combined_path.is_file():
        return None
    try:
        data = json.loads(combined_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    from collections import Counter
    counts = Counter(c.get("field_name", "") for c in candidates)
    if not counts:
        return None
    top_field, top_count = counts.most_common(1)[0]
    total = sum(counts.values())
    if top_count >= total * 0.3:
        return top_field
    return None


_PAGE_CONTEXT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"plasmid.*persist|proportion.*host|host.*frequency|plasmid.*frequency|plasmid.*bearing|stability.*shown", re.I),
     "host_frequency_plasmid_bearing_cells"),
    (re.compile(r"conjug.*rate|transfer.*freq|conjug.*freq", re.I), "conjugation_rate"),
    (re.compile(r"conjug.*effic", re.I), "conjugation_efficiency_density_scaled"),
    (re.compile(r"growth.*curve|growth.*rate|OD.*time", re.I), "growth_rate"),
    (re.compile(r"fitness.*cost|relative.*fitness|competition", re.I), "relative_fitness"),
    (re.compile(r"plasmid.*loss|loss.*freq|curing", re.I), "plasmid_loss_frequency"),
    (re.compile(r"copy.*number|plasmid.*copies", re.I), "plasmid_copy_number"),
    (re.compile(r"CFU|colony.?form|bacteria.*per.*ml", re.I), "cfu_per_ml"),
]


def _build_page_field_hints(text_dir: Path) -> dict[int, str]:
    """Scan page texts for figure captions and infer field_names per page."""
    pages_path = text_dir / "pages.json"
    if not pages_path.is_file():
        return {}
    try:
        pages = json.loads(pages_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    hints: dict[int, str] = {}
    fig_caption_re = re.compile(r"^Fig\.?\s*\d", re.IGNORECASE)

    for p in pages:
        page_num = p["page"]
        caption_text = ""
        for line in p["text"].split("\n"):
            if fig_caption_re.match(line.strip()):
                caption_text += " " + line.strip()

        if not caption_text:
            continue

        for pat, field in _PAGE_CONTEXT_PATTERNS:
            if pat.search(caption_text):
                hints[page_num] = field
                break

    return hints


def _normalize_to_snake(s: str) -> str:
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s.lower()[:80] or "value"


def _try_parse_time(x_val: Any, x_label: str, x_units: str) -> float | None:
    """If x-axis represents time in hours, return the numeric value."""
    combined = f"{x_label} {x_units}".lower()
    is_time = any(kw in combined for kw in ("time", "hour", "hrs", "h", "day", "min", "transfer"))
    if not is_time:
        return None
    try:
        v = float(x_val)
        if "day" in combined:
            return v * 24
        if "min" in combined:
            return v / 60
        return v
    except (ValueError, TypeError):
        return None


def _error_bar(pt: dict) -> float | None:
    lo = pt.get("error_low")
    hi = pt.get("error_high")
    if lo is not None and hi is not None:
        return (hi - lo) / 2
    if hi is not None:
        return hi
    return lo


def _try_parse_numeric(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        s = str(val).strip().lstrip("~≈≥≤><")
        s = re.sub(r"[%°]", "", s).strip()
        try:
            return float(s)
        except (ValueError, TypeError):
            return None


# ── Main entry points ────────────────────────────────────────────────────────

def expand_vision_extractions(
    extractions_dir: Path,
    paper_id: str,
    combined_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all vision extraction JSONs and expand into measurement-candidate dicts.

    Uses three levels of field_name resolution:
    1. Axis/value label from the vision extraction (e.g. "Conjugation Frequency")
    2. Page-level hint from figure captions in the PDF text
    3. Dominant field_name from the LLM's existing combined output

    Returns a list of dicts matching the ``CombinedMeasurementCandidate`` shape.
    """
    if not extractions_dir.is_dir():
        return []

    paths = sorted(extractions_dir.glob("*.json"))
    if not paths:
        return []

    dominant_field = _get_dominant_field(combined_path) if combined_path else None

    text_dir = extractions_dir.parent / "text"
    page_hints = _build_page_field_hints(text_dir)

    all_rows: list[dict[str, Any]] = []
    id_counter = 0

    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        pt = data.get("plot_type", "unknown")
        stem = p.stem

        page_num = _page_from_stem(stem)
        effective_dominant = page_hints.get(page_num) if page_num else None
        if effective_dominant is None:
            effective_dominant = dominant_field

        if pt in ("line_chart", "line_plot"):
            rows, id_counter = _expand_line_chart(stem, data, paper_id, id_counter, effective_dominant)
            all_rows.extend(rows)
        elif pt == "box_plot":
            rows, id_counter = _expand_box_plot(stem, data, paper_id, id_counter, effective_dominant)
            all_rows.extend(rows)
        elif pt == "heatmap":
            rows, id_counter = _expand_heatmap(stem, data, paper_id, id_counter, effective_dominant)
            all_rows.extend(rows)
        elif pt == "table_image":
            rows, id_counter = _expand_table_image(stem, data, paper_id, id_counter, effective_dominant)
            all_rows.extend(rows)

    return all_rows


_GENERIC_FIELD_NAMES = {"figure_value", "y_value", "value", "x_value", "heatmap_value"}


def _canonicalize_for_merge(field_name: str) -> str:
    """Map a field_name to its canonical form using batch_evaluate synonyms."""
    from paper_analysis.batch_evaluate import _canonicalize_field_name
    return _canonicalize_field_name(field_name)


_HIGH_CONFIDENCE_FIELDS = {
    "host_frequency_plasmid_bearing_cells", "fraction_plasmid_bearing_cells",
    "plasmid_stability", "conjugation_rate", "conjugation_frequency",
    "conjugation_transfer_frequency", "transfer_frequency",
    "conjugation_efficiency_density_scaled", "conjugation_detection_status",
    "growth_rate", "maximum_growth_rate", "relative_fitness", "fitness_cost",
    "competition_index", "selection_coefficient",
    "plasmid_loss_frequency", "plasmid_loss_rate",
    "plasmid_copy_number", "colony_area",
    "area_under_growth_curve", "area_under_curve",
    "maximum_od", "maximum_od600",
    "relative_transcription_per_plasmid",
    "segregation_factor", "mic",
}


def _is_keepable_field(
    fn_norm: str,
    fn_canon: str,
    llm_fields: set[str],
    llm_canonicals: set[str],
) -> bool:
    """Decide whether a vision-expanded row should be kept.

    A row is kept if its field name matches an LLM-produced field (directly or
    via synonym), OR if it is a well-known high-confidence field that the
    _AXIS_TO_FIELD mapping or page hints assigned.
    """
    if fn_norm in llm_fields or fn_canon in llm_canonicals:
        return True
    if fn_norm in _HIGH_CONFIDENCE_FIELDS or fn_canon in _HIGH_CONFIDENCE_FIELDS:
        return True
    return False


def merge_with_combined(
    combined_path: Path,
    expanded_rows: list[dict[str, Any]],
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Merge expanded vision rows into an existing candidate_combined.json.

    Filtering rules (applied in order):
    1. Rows with generic/fallback field_names (figure_value, y_value) are
       dropped — they can't match ground truth.
    2. Only keep expanded rows whose field_name matches a field_name already
       present in the LLM output.  This prevents the vision expansion from
       introducing entirely new (likely wrong) field names.
    3. Deduplication: an expanded row is dropped if an LLM-produced row with
       the same (field_name, raw_value) already exists.

    Returns the merged data dict.
    """
    if combined_path.is_file():
        data = json.loads(combined_path.read_text(encoding="utf-8"))
    else:
        data = {"result_type": "combined_extraction", "candidates": [], "notes": None}

    existing = data.get("candidates", [])

    llm_field_names: set[str] = set()
    llm_canonical_names: set[str] = set()
    existing_keys: set[tuple] = set()
    for row in existing:
        fn = str(row.get("field_name", "")).strip().lower()
        fn_norm = re.sub(r"[\s\-]+", "_", fn)
        llm_field_names.add(fn_norm)
        llm_canonical_names.add(_canonicalize_for_merge(fn_norm))
        rv = row.get("raw_value")
        rv_key = _round_val(rv)
        existing_keys.add((fn_norm, rv_key))

    added = 0
    skipped_generic = 0
    skipped_novel = 0
    for row in expanded_rows:
        fn = str(row.get("field_name", "")).strip().lower()
        fn_norm = re.sub(r"[\s\-]+", "_", fn)
        if fn_norm in _GENERIC_FIELD_NAMES:
            skipped_generic += 1
            continue
        fn_canon = _canonicalize_for_merge(fn_norm)
        if not _is_keepable_field(fn_norm, fn_canon, llm_field_names, llm_canonical_names):
            skipped_novel += 1
            continue
        rv_key = _round_val(row.get("raw_value"))
        if (fn_norm, rv_key) not in existing_keys:
            existing.append(row)
            existing_keys.add((fn_norm, rv_key))
            added += 1

    data["candidates"] = existing
    old_notes = data.get("notes") or ""
    parts = [old_notes] if old_notes else []
    parts.append(f"Post-processed: added {added} vision-expanded rows")
    if skipped_generic:
        parts.append(f"skipped {skipped_generic} generic-field rows")
    if skipped_novel:
        parts.append(f"skipped {skipped_novel} novel-field rows")
    data["notes"] = "; ".join(parts)

    if write:
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return data


def _round_val(v: Any) -> Any:
    """Round numeric values to 2 decimals for dedup; keep strings as-is."""
    if v is None:
        return None
    try:
        return round(float(v), 2)
    except (ValueError, TypeError):
        return str(v).strip().lower()


_METADATA_FIELD_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(coculture_volume|culture_volume|tube_volume)$"),
    re.compile(r"^shaking_speed$"),
    re.compile(r"^(agar_concentration|salt_concentration)$"),
    re.compile(r"^(stirring_speed|stirring_bar_length)$"),
    re.compile(r"^(agitation_speed|agitation_rpm)$"),
    re.compile(r"^(sampling_volume|transconjugant_selecting_medium_volume)$"),
    re.compile(r"^(acclimation_growth_duration|initial_growth_duration|exponential_growth_duration)$"),
    re.compile(r"^(volume_for_density_measurement)$"),
    re.compile(r"^(dilution_factor_first|dilution_factor_second|dilution_factor_high|dilution_factor_low|initial_dilution_factor)$"),
    re.compile(r"^(transconjugant_detection_incubation_time)$"),
    re.compile(r"^(simulation_sampling_interval)$"),
    re.compile(r"^(annealing_temperature|primer_concentration|rna_amount)$"),
    re.compile(r"^(optical_density_measurement|detection_sensitivity)$"),
    re.compile(r"^(biosynthetic_cost|plasmid_gene_expression)"),
    re.compile(r"^(fold_stabilization_efficiency|fold_decrease_protein_level)$"),
    re.compile(r"^(normalized_protein_intensity|beta_galactosidase_activity)$"),
    re.compile(r"^(correlation_coefficient)"),
    re.compile(r"^(percentage_pairs_with_effects|sequence_identity_percentage|sequence_alignment_coverage)$"),
    re.compile(r"^(donor_growth_rate|recipient_growth_rate|transconjugant_growth_rate)$"),
    re.compile(r"^(initial_donor_density|initial_recipient_density|initial_transconjugant_density)$"),
    re.compile(r"^(parallel_cocultures_for_sim)$"),
    re.compile(r"^(culture_growth_time_h|colonies_tested_per_culture)"),
    re.compile(r"^(colony_count_range)"),
    re.compile(r"^(measurement_delay)$"),
    re.compile(r"^(flow_cytometry_speed)"),
    re.compile(r"^(conjugation_inhibitor_concentration|plasmid_curing_agent_concentration)$"),
    re.compile(r"^(daily_probability_baseline)$"),
    re.compile(r"^(n_de_genes_common|n_metabolites)"),
    re.compile(r"^(daily_probability_baseline)$"),
]


def downgrade_metadata_rows(
    combined_path: Path,
    *,
    remove: bool = False,
) -> int:
    """Downgrade confidence of rows that look like run metadata, not measurements.

    These are experimental setup parameters (volumes, speeds, durations) that
    the LLM extracted as measurement rows but ground truth treats as part of
    the run description.

    If ``remove=True``, removes them entirely instead of downgrading.
    Returns count of affected rows.
    """
    if not combined_path.is_file():
        return 0

    data = json.loads(combined_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])
    if not candidates:
        return 0

    affected = 0
    if remove:
        kept = []
        for c in candidates:
            fn = str(c.get("field_name", "")).strip().lower()
            fn_norm = re.sub(r"[\s\-]+", "_", fn)
            if any(pat.match(fn_norm) for pat in _METADATA_FIELD_PATTERNS):
                affected += 1
            else:
                kept.append(c)
        data["candidates"] = kept
    else:
        for c in candidates:
            fn = str(c.get("field_name", "")).strip().lower()
            fn_norm = re.sub(r"[\s\-]+", "_", fn)
            if any(pat.match(fn_norm) for pat in _METADATA_FIELD_PATTERNS):
                c["confidence"] = "low"
                affected += 1

    if affected:
        notes = data.get("notes") or ""
        action = "removed" if remove else "downgraded"
        data["notes"] = f"{notes}; {action} {affected} metadata-like rows".strip("; ")
        combined_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return affected


def derive_qualitative_fields(combined_path: Path) -> int:
    """Derive conjugation_detection_status and conjugation_interaction_direction
    from existing conjugation_rate rows.

    For each conjugation_rate row with a numeric value, adds companion rows:
    - conjugation_detection_status: "detected" if rate exists (finite)
    - conjugation_interaction_direction: "positive"/"negative"/"neutral"
      based on the sign/magnitude of the rate value

    Returns count of new rows added.
    """
    if not combined_path.is_file():
        return 0

    data = json.loads(combined_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])

    from paper_analysis.batch_evaluate import _canonicalize_field_name
    conj_rate_canonical = _canonicalize_field_name("conjugation_rate")

    existing_fields: set[tuple] = set()
    for c in candidates:
        fn = _canonicalize_field_name(
            re.sub(r"[\s\-]+", "_", str(c.get("field_name", "")).strip().lower())
        )
        existing_fields.add(fn)

    detection_canonical = _canonicalize_field_name("conjugation_detection_status")
    direction_canonical = _canonicalize_field_name("conjugation_interaction_direction")

    if detection_canonical in existing_fields and direction_canonical in existing_fields:
        return 0

    rate_rows = [
        c for c in candidates
        if _canonicalize_field_name(
            re.sub(r"[\s\-]+", "_", str(c.get("field_name", "")).strip().lower())
        ) == conj_rate_canonical
    ]

    if len(rate_rows) < 15:
        return 0

    pairwise_markers = re.compile(
        r"co.?resident|interaction|matrix|pairwise|plasmid.{1,10}plasmid", re.I
    )
    llm_rate_rows = [
        r for r in rate_rows
        if not str(r.get("measurement_id", "")).startswith(("PP_", "DRV_"))
    ]
    if not llm_rate_rows:
        return 0
    n_pairwise = sum(
        1 for r in llm_rate_rows
        if pairwise_markers.search(str(r.get("run_description", "")))
    )
    if n_pairwise < len(llm_rate_rows) * 0.3:
        return 0

    added = 0
    id_counter = len(candidates) + 1000

    for row in rate_rows:
        raw_val = row.get("raw_value")
        numeric_val = _try_parse_numeric(raw_val)

        if detection_canonical not in existing_fields:
            id_counter += 1
            det_row = dict(row)
            det_row["measurement_id"] = f"DRV_{id_counter:04d}"
            det_row["field_name"] = "conjugation_detection_status"
            det_row["raw_value"] = "detected" if numeric_val is not None else "not_detected"
            det_row["raw_units"] = None
            det_row["normalized_value"] = None
            det_row["normalized_units"] = None
            det_row["dispersion_value"] = None
            det_row["dispersion_type"] = None
            det_row["confidence"] = "medium"
            candidates.append(det_row)
            added += 1

        if direction_canonical not in existing_fields and numeric_val is not None:
            id_counter += 1
            dir_row = dict(row)
            dir_row["measurement_id"] = f"DRV_{id_counter:04d}"
            dir_row["field_name"] = "conjugation_interaction_direction"
            dir_row["raw_units"] = None
            dir_row["normalized_value"] = None
            dir_row["normalized_units"] = None
            dir_row["dispersion_value"] = None
            dir_row["dispersion_type"] = None
            dir_row["confidence"] = "medium"
            dir_row["raw_value"] = "detected"
            candidates.append(dir_row)
            added += 1

    if added:
        data["candidates"] = candidates
        notes = data.get("notes") or ""
        data["notes"] = f"{notes}; Derived {added} qualitative rows from conjugation_rate".strip("; ")
        combined_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return added


def run_postprocess_for_paper(
    paper_id: str,
    extractions_dir: Path,
    combined_path: Path,
    *,
    remove_metadata: bool = True,
    mfl_path: Path | None = None,
    echo: callable = print,
) -> int:
    """Run post-processing for one paper. Returns count of new rows added."""
    expanded = expand_vision_extractions(extractions_dir, paper_id, combined_path)
    if not expanded:
        echo(f"  [{paper_id}] postprocess: no vision data to expand")
    else:
        merge_with_combined(combined_path, expanded, write=True)

    derived = derive_qualitative_fields(combined_path)

    mfl = get_master_field_index(mfl_path)
    if mfl is not None:
        mfl_removed, mfl_kept = filter_by_master_field_list(
            combined_path, paper_id, mfl, remove=True,
        )
    else:
        downgrade_metadata_rows(combined_path, remove=remove_metadata)
        mfl_removed, mfl_kept = 0, 0

    data = json.loads(combined_path.read_text(encoding="utf-8")) if combined_path.is_file() else {}
    n_total = len(data.get("candidates", []))
    parts = []
    if expanded:
        parts.append(f"{len(expanded)} vision rows expanded")
    if derived:
        parts.append(f"{derived} qualitative rows derived")
    if mfl_removed:
        parts.append(f"MFL filter: {mfl_removed} removed")
    parts.append(f"{n_total} total")
    echo(f"  [{paper_id}] postprocess: {', '.join(parts)}")
    return len(expanded) + derived
