"""Batch processing: iterate over a directory of papers and run the full pipeline on each."""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from paper_analysis.combined_analyze_llm import run_combined_analysis
from paper_analysis.config import (
    BatchConfig,
    PocConfig,
    effective_text_config,
    resolve_figure_targets,
)
from paper_analysis.pdf_discovery import discover_figure_targets
from paper_analysis.pdf_figures import extract_all_figures
from paper_analysis.pdf_text_tables import (
    build_figure_summary,
    build_llm_context,
    load_text_artifacts,
    run_text_dump,
)
from paper_analysis.postprocess import run_postprocess_for_paper
from paper_analysis.vision.base import build_vision_client


@dataclass
class PaperResult:
    paper_id: str
    pdf_path: Path
    n_figures: int = 0
    n_extractions: int = 0
    n_combined_rows: int = 0
    error: str | None = None


@dataclass
class BatchResult:
    papers: list[PaperResult] = field(default_factory=list)

    @property
    def succeeded(self) -> list[PaperResult]:
        return [p for p in self.papers if p.error is None]

    @property
    def failed(self) -> list[PaperResult]:
        return [p for p in self.papers if p.error is not None]

    def summary(self) -> str:
        lines = [f"Batch complete: {len(self.succeeded)}/{len(self.papers)} papers succeeded"]
        total_rows = sum(p.n_combined_rows for p in self.succeeded)
        lines.append(f"Total combined rows: {total_rows}")
        if self.failed:
            lines.append("Failed papers:")
            for p in self.failed:
                lines.append(f"  {p.paper_id}: {p.error}")
        return "\n".join(lines)


def list_pdfs(batch_cfg: BatchConfig) -> list[tuple[Path, str]]:
    """Return sorted list of (pdf_path, paper_id) for all PDFs in papers_dir."""
    papers_dir = batch_cfg.papers_dir
    if not papers_dir.is_dir():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {papers_dir}")
    return [(p, batch_cfg.paper_id_for(p.name)) for p in pdfs]


def run_pipeline_for_paper(
    batch_cfg: BatchConfig,
    pdf_path: Path,
    paper_id: str,
    *,
    skip_vision: bool = False,
    echo: callable = print,
) -> PaperResult:
    """Run the full extract-text -> figures -> vision -> combined pipeline for one paper."""
    result = PaperResult(paper_id=paper_id, pdf_path=pdf_path)
    cfg = batch_cfg.poc_config_for_paper(pdf_path, paper_id)

    try:
        # 1. Extract text + native tables
        tc = effective_text_config(cfg)
        echo(f"  [{paper_id}] extract-text ...")
        run_text_dump(cfg.pdf, tc)

        # 2. Discover + extract figures (increase recursion limit for complex PDFs)
        targets = []
        if cfg.auto_discover and cfg.auto_discover.enabled:
            echo(f"  [{paper_id}] discover-bboxes ...")
            old_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(max(old_limit, 10000))
                targets = resolve_figure_targets(cfg)
            except RecursionError:
                echo(f"  [{paper_id}] discover-bboxes hit recursion limit, skipping figures")
                targets = []
            finally:
                sys.setrecursionlimit(old_limit)
            result.n_figures = len(targets)
            if targets:
                echo(f"  [{paper_id}] extract-figures ({len(targets)} targets) ...")
                extract_all_figures(cfg.pdf, targets, cfg.artifacts)

        # 3. Run vision on each crop
        if targets and not skip_vision:
            provider = cfg.vision.provider or os.environ.get(
                "PAPER_ANALYSIS_VISION_PROVIDER", "anthropic"
            )
            client = build_vision_client(provider=provider, model=cfg.vision.model)
            ext_dir = cfg.artifacts.extractions_dir
            ext_dir.mkdir(parents=True, exist_ok=True)
            fig_dir = cfg.artifacts.figures_dir
            n_ext = 0
            for fig in targets:
                png_path = fig_dir / f"{fig.id}.png"
                if not png_path.is_file():
                    continue
                data = png_path.read_bytes()
                try:
                    extraction = client.extract_figure(data, fig.plot_type, max_retries=1)
                    dest = ext_dir / f"{fig.id}.json"
                    dest.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
                    n_ext += 1
                except Exception:
                    echo(f"  [{paper_id}] vision failed for {fig.id}: {traceback.format_exc()}")
            result.n_extractions = n_ext
            echo(f"  [{paper_id}] run-vision ({n_ext} extractions)")

        # 4. Combined extraction
        echo(f"  [{paper_id}] extract-combined ...")
        pages, tables = load_text_artifacts(tc)
        content = build_llm_context(pages, tables, tc.max_context_chars)
        fig_summary = build_figure_summary(cfg.artifacts.extractions_dir)
        if fig_summary:
            content = content + "\n\n" + fig_summary

        provider = cfg.vision.provider or os.environ.get(
            "PAPER_ANALYSIS_VISION_PROVIDER", "anthropic"
        )
        model = tc.llm_model or cfg.vision.model
        batch_out = run_combined_analysis(content, provider=provider, model=model, max_retries=2)
        if not batch_out.candidates:
            echo(f"  [{paper_id}] got 0 candidates, retrying...")
            batch_out = run_combined_analysis(content, provider=provider, model=model, max_retries=2)

        out_path = tc.output_dir / "candidate_combined.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(batch_out.model_dump_json(indent=2), encoding="utf-8")

        # 5. Post-process: expand vision data into individual measurement rows
        ext_dir = cfg.artifacts.extractions_dir
        if ext_dir.is_dir() and any(ext_dir.glob("*.json")):
            run_postprocess_for_paper(
                paper_id, ext_dir, out_path,
                mfl_path=batch_cfg.field_list_path,
                echo=echo,
            )
            merged = json.loads(out_path.read_text(encoding="utf-8"))
            result.n_combined_rows = len(merged.get("candidates", []))
        else:
            result.n_combined_rows = len(batch_out.candidates)

        n_runs = len({c.get("run_id", "") for c in json.loads(out_path.read_text(encoding="utf-8")).get("candidates", [])})
        echo(
            f"  [{paper_id}] done: {result.n_combined_rows} measurements, {n_runs} runs"
        )

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        echo(f"  [{paper_id}] FAILED: {result.error}")

    return result


def run_batch(
    batch_cfg: BatchConfig,
    *,
    papers: list[str] | None = None,
    skip_vision: bool = False,
    echo: callable = print,
) -> BatchResult:
    """Run the pipeline on all (or selected) papers."""
    all_pdfs = list_pdfs(batch_cfg)
    if papers:
        filter_set = set(papers)
        all_pdfs = [(p, pid) for p, pid in all_pdfs if pid in filter_set or p.stem in filter_set]
        if not all_pdfs:
            raise ValueError(f"No PDFs matched filter: {papers}")

    echo(f"Processing {len(all_pdfs)} papers...")
    batch_result = BatchResult()
    for pdf_path, paper_id in all_pdfs:
        echo(f"\n--- {paper_id} ({pdf_path.name}) ---")
        pr = run_pipeline_for_paper(
            batch_cfg, pdf_path, paper_id, skip_vision=skip_vision, echo=echo
        )
        batch_result.papers.append(pr)

    echo(f"\n{batch_result.summary()}")
    return batch_result


def merge_combined_outputs(batch_cfg: BatchConfig) -> list[dict]:
    """Load all per-paper candidate_combined.json and merge into one list."""
    all_rows: list[dict] = []
    papers_dir = batch_cfg.papers_dir
    if not papers_dir.is_dir():
        return all_rows
    for pdf_path in sorted(papers_dir.glob("*.pdf")):
        paper_id = batch_cfg.paper_id_for(pdf_path.name)
        combined_path = batch_cfg.artifacts_dir / paper_id / "text" / "candidate_combined.json"
        if not combined_path.is_file():
            continue
        try:
            data = json.loads(combined_path.read_text(encoding="utf-8"))
            for c in data.get("candidates", []):
                c["_paper_id_mapped"] = paper_id
                all_rows.append(c)
        except (json.JSONDecodeError, OSError):
            continue
    return all_rows
