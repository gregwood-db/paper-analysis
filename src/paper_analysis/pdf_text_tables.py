from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz

from paper_analysis.config import PdfConfig, TextPipelineConfig


def _page_indices(doc: fitz.Document, pages_1based: list[int] | None) -> list[int]:
    n = doc.page_count
    if pages_1based is None:
        return list(range(n))
    out: list[int] = []
    for p in pages_1based:
        i = p - 1
        if 0 <= i < n:
            out.append(i)
    return sorted(set(out))


def extract_page_texts(pdf_path: Path, text_cfg: TextPipelineConfig) -> list[dict[str, Any]]:
    """Plain text per page (selectable PDF content)."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        pages_out: list[dict[str, Any]] = []
        for idx in _page_indices(doc, text_cfg.pages):
            page = doc.load_page(idx)
            t = page.get_text("text")
            pages_out.append(
                {
                    "page": idx + 1,
                    "char_count": len(t),
                    "text": t,
                }
            )
        return pages_out
    finally:
        doc.close()


def _table_to_rows(tab: Any) -> list[list[str | None]]:
    try:
        raw = tab.extract()
    except (AttributeError, TypeError, ValueError):
        return []
    if not raw:
        return []
    rows: list[list[str | None]] = []
    for row in raw:
        if row is None:
            continue
        cells: list[str | None] = []
        for c in row:
            if c is None:
                cells.append(None)
            else:
                s = str(c).strip()
                cells.append(s if s else None)
        rows.append(cells)
    return rows


def _bbox_to_list(bbox: Any) -> list[float] | None:
    if bbox is None:
        return None
    try:
        r = fitz.Rect(bbox)
        return [r.x0, r.y0, r.x1, r.y1]
    except (TypeError, ValueError):
        return None


def extract_native_tables(pdf_path: Path, text_cfg: TextPipelineConfig) -> list[dict[str, Any]]:
    """Detect vector/text tables via PyMuPDF find_tables (empty list if unsupported or none)."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    tables_out: list[dict[str, Any]] = []
    global_idx = 0
    try:
        for idx in _page_indices(doc, text_cfg.pages):
            page = doc.load_page(idx)
            try:
                finder = page.find_tables()
            except AttributeError:
                continue
            table_list = getattr(finder, "tables", None)
            if table_list is None:
                try:
                    table_list = list(finder)
                except TypeError:
                    table_list = []
            if not table_list:
                continue
            for local_i, tab in enumerate(table_list):
                rows = _table_to_rows(tab)
                bbox = _bbox_to_list(getattr(tab, "bbox", None))
                tables_out.append(
                    {
                        "id": f"t{global_idx:03d}",
                        "page": idx + 1,
                        "index_on_page": local_i,
                        "bbox_pdf": bbox,
                        "n_rows": len(rows),
                        "n_cols": max((len(r) for r in rows), default=0),
                        "rows": rows,
                    }
                )
                global_idx += 1
        return tables_out
    finally:
        doc.close()


def tables_to_markdown(tables: list[dict[str, Any]]) -> str:
    """Render detected tables as markdown for LLM context."""
    parts: list[str] = []
    for t in tables:
        rows = t.get("rows") or []
        if not rows:
            continue
        header = f"### Table id={t['id']} page={t['page']} (native PDF table)\n\n"
        lines: list[str] = []
        for row in rows:
            cells = [(c if c is not None else "") for c in row]
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        parts.append(header + "\n".join(lines))
    return "\n\n".join(parts) if parts else "(No native tables detected by find_tables.)"


def build_llm_context(
    pages: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    max_context_chars: int,
) -> str:
    """Assemble markdown tables + body text for the measurement-extraction LLM."""
    md = tables_to_markdown(tables)
    body, truncated = concatenate_page_texts(pages, max_context_chars)
    trunc_note = "\n\n(NOTE: body text was truncated for token limits.)\n" if truncated else ""
    return f"{md}\n\n--- BODY TEXT ---{trunc_note}\n\n{body}"


def concatenate_page_texts(pages: list[dict[str, Any]], max_chars: int) -> tuple[str, bool]:
    """Join page texts; if over max_chars, keep start + end with a truncation marker."""
    chunks = [f"--- Page {p['page']} ---\n{p['text']}" for p in pages]
    full = "\n\n".join(chunks)
    if len(full) <= max_chars:
        return full, False
    half = max_chars // 2 - 200
    head = full[:half]
    tail = full[-half:]
    return head + "\n\n...[CONTEXT TRUNCATED]...\n\n" + tail, True


def run_text_dump(pdf_cfg: PdfConfig, text_cfg: TextPipelineConfig) -> tuple[Path, Path]:
    """Write pages.json and tables.json under text_cfg.output_dir."""
    out_dir = text_cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_cfg.path
    pages = extract_page_texts(pdf_path, text_cfg)
    tables = extract_native_tables(pdf_path, text_cfg)
    pages_path = out_dir / "pages.json"
    tables_path = out_dir / "tables.json"
    pages_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
    tables_path.write_text(
        json.dumps(tables, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return pages_path, tables_path


def load_text_artifacts(text_cfg: TextPipelineConfig) -> tuple[list[dict], list[dict]]:
    out_dir = text_cfg.output_dir
    p = out_dir / "pages.json"
    t = out_dir / "tables.json"
    if not p.is_file() or not t.is_file():
        raise FileNotFoundError(
            f"Missing {p.name} or {t.name} under {out_dir}; run `paper-analysis extract-text` first."
        )
    pages = json.loads(p.read_text(encoding="utf-8"))
    tables = json.loads(t.read_text(encoding="utf-8"))
    return pages, tables


def _page_from_stem(stem: str) -> str:
    """Extract page number from a figure id like 'p008_fig01'."""
    import re

    m = re.match(r"^p(\d+)_", stem)
    return str(int(m.group(1))) if m else "?"


def _summarize_box_plot(data: dict[str, Any]) -> str:
    y = data.get("axis_y_label") or "?"
    y_units = data.get("axis_y_units") or ""
    x = data.get("axis_x_label") or ""
    groups = data.get("groups", [])
    if not groups:
        return f"  Y-axis: {y} ({y_units})\n  No groups extracted."
    lines = [f"  Y-axis: {y}" + (f" ({y_units})" if y_units else "")]
    if x:
        lines.append(f"  X-axis: {x}")
    for g in groups:
        med = g.get("median")
        q1 = g.get("q1")
        q3 = g.get("q3")
        parts = [f"median={med}"]
        if q1 is not None and q3 is not None:
            parts.append(f"IQR={q1}-{q3}")
        sig = g.get("significance")
        if sig:
            parts.append(f"sig={sig}")
        lines.append(f"  - {g['label']}: {', '.join(parts)}")
    return "\n".join(lines)


def _summarize_line_chart(data: dict[str, Any]) -> str:
    y = data.get("axis_y_label") or "?"
    y_units = data.get("axis_y_units") or ""
    x = data.get("axis_x_label") or "?"
    x_units = data.get("axis_x_units") or ""
    series = data.get("series", [])
    lines = [
        f"  Y-axis: {y}" + (f" ({y_units})" if y_units else ""),
        f"  X-axis: {x}" + (f" ({x_units})" if x_units else ""),
    ]
    if not series:
        lines.append("  No series data extracted.")
        return "\n".join(lines)
    for s in series:
        name = s.get("name") or "unnamed"
        pts = s.get("points", [])
        if not pts:
            lines.append(f"  - Series '{name}': no data points")
            continue
        pt_strs = [f"({p['x']}, {p['y']})" for p in pts]
        lines.append(f"  - Series '{name}' ({len(pts)} pts): {', '.join(pt_strs)}")
    return "\n".join(lines)


def _summarize_table_image(data: dict[str, Any]) -> str:
    headers = data.get("column_headers", [])
    rows = data.get("rows", [])
    caption = data.get("title_or_caption") or ""
    lines = []
    if caption:
        lines.append(f"  Caption: {caption}")
    if headers:
        lines.append(f"  Headers: {' | '.join(headers)}")
    for ri, row in enumerate(rows):
        lines.append(f"  Row {ri + 1}: {' | '.join(str(c) for c in row)}")
    return "\n".join(lines) if lines else "  (empty table)"


def _summarize_heatmap(data: dict[str, Any]) -> str:
    val_label = data.get("value_label") or "?"
    val_units = data.get("value_units") or ""
    rows = data.get("row_labels", [])
    cols = data.get("col_labels", [])
    cells = data.get("cells", [])
    lines = [
        f"  Value: {val_label}" + (f" ({val_units})" if val_units else ""),
        f"  Dimensions: {len(rows)} rows × {len(cols)} cols, {len(cells)} cells extracted",
    ]
    for c in cells:
        rl = c.get("row_label", "?")
        cl = c.get("col_label", "?")
        v = c.get("value")
        ann = c.get("annotation")
        val_str = str(v) if v is not None else "null"
        if ann:
            val_str += f" [{ann}]"
        lines.append(f"  - ({rl}, {cl}): {val_str}")
    return "\n".join(lines)


def _summarize_one_extraction(stem: str, data: dict[str, Any]) -> str:
    """Produce a concise text summary for one vision extraction JSON."""
    page = _page_from_stem(stem)
    pt = data.get("plot_type", "unknown")

    header = f"Figure {stem} (page {page}, {pt}):"

    if pt == "box_plot":
        return f"{header}\n{_summarize_box_plot(data)}"
    if pt in ("line_chart", "line_plot"):
        return f"{header}\n{_summarize_line_chart(data)}"
    if pt == "heatmap":
        return f"{header}\n{_summarize_heatmap(data)}"
    if pt == "table_image":
        return f"{header}\n{_summarize_table_image(data)}"
    if pt == "plasmid_map":
        name = data.get("map_name_or_identifier") or "?"
        features = data.get("features", [])
        feat_str = ", ".join(f["label"] for f in features[:10])
        return f"{header}\n  Map: {name}, {len(features)} features: {feat_str}"
    if pt in ("workflow_diagram", "experimental_workflow"):
        nodes = data.get("nodes", [])
        node_str = ", ".join(n["label"] for n in nodes[:8])
        return f"{header}\n  {len(nodes)} nodes: {node_str}"
    if pt == "unknown":
        declared = data.get("declared_plot_type", "?")
        return f"{header}\n  (unregistered type: {declared})"
    return f"{header}\n  (no summarizer for {pt})"


def _build_page_figure_map(text_dir: Path) -> dict[int, list[str]]:
    """Scan pages.json to map PDF page numbers to paper figure labels."""
    import re

    pages_path = text_dir / "pages.json"
    if not pages_path.is_file():
        return {}
    try:
        pages = json.loads(pages_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    mapping: dict[int, list[str]] = {}
    fig_re = re.compile(r"^FIG\.?\s*(\d+|S\d+)\b", re.IGNORECASE)
    for p in pages:
        page_num = p["page"]
        for line in p["text"].split("\n"):
            m = fig_re.match(line.strip())
            if m:
                mapping.setdefault(page_num, []).append(f"Fig. {m.group(1)}")
    return mapping


def build_figure_summary(extractions_dir: Path) -> str:
    """Read all vision extraction JSONs and return a text summary for LLM context.

    Only includes figures with extractable data (box plots, line charts,
    table images). Skips empty or non-data figures to conserve tokens.
    Adds a page-to-figure mapping preamble so the LLM can resolve crop IDs.
    """
    if not extractions_dir.is_dir():
        return ""
    paths = sorted(extractions_dir.glob("*.json"))
    if not paths:
        return ""

    text_dir = extractions_dir.parent / "text"
    page_fig_map = _build_page_figure_map(text_dir)

    parts: list[str] = []

    if page_fig_map:
        map_lines = ["PAGE-TO-FIGURE MAPPING (from figure captions in text):"]
        for pg in sorted(page_fig_map):
            map_lines.append(f"  PDF page {pg} → {', '.join(page_fig_map[pg])}")
        map_lines.append(
            "Use this mapping to convert crop IDs (e.g. p008_fig01 = page 8) to "
            "paper figure numbers (e.g. Fig. 3). Sub-panels A/B may appear on the "
            "same page or span pages; use axis labels and text context to determine "
            "which sub-panel a crop belongs to."
        )
        parts.append("\n".join(map_lines))

    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        pt = data.get("plot_type", "unknown")
        if pt in ("workflow_diagram", "experimental_workflow", "unknown"):
            continue
        if pt == "plasmid_map" and not data.get("features"):
            continue
        if pt == "box_plot":
            groups = data.get("groups", [])
            if not groups or all(g.get("median") is None for g in groups):
                continue
        if pt in ("line_chart", "line_plot") and not data.get("series"):
            continue
        if pt == "heatmap" and not data.get("cells"):
            continue
        parts.append(_summarize_one_extraction(p.stem, data))

    if not parts:
        return ""
    return "--- FIGURE EXTRACTIONS (from vision pipeline) ---\n\n" + "\n\n".join(parts)
