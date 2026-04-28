from __future__ import annotations

COMBINED_SYSTEM = """\
You extract structured experimental data from scientific papers, producing a \
denormalized table where every measurement row carries its full experimental \
run metadata. Reply with a single JSON object only — no markdown fences, \
no commentary. Use null for unknown values."""

COMBINED_USER = """\
You are given three types of input from a scientific paper:
(1) Native PDF tables serialized as markdown.
(2) Body text from the paper.
(3) Figure extraction data — structured numeric values already read from \
figure panels by a vision model (box-plot medians/IQR, line-chart points, \
table-image cells). These appear under "--- FIGURE EXTRACTIONS ---".

Your task has TWO phases executed together in ONE output:

Phase 1 — Identify every distinct experimental **run** (a unique combination \
of experiment type + strain/organism + treatment/plasmid/condition). Assign \
each a stable Run_ID (RUN_001, RUN_002, …). Controls, negative controls, \
and evolved/mutant variants each get their own Run_ID.

Phase 2 — For every quantitative **measurement** reported for each run, emit \
one row that carries the full run metadata *plus* the measurement data. Assign \
each row a globally sequential Measurement_ID (MEAS_001, MEAS_002, …). When a \
single run produces multiple measured outcomes (e.g. AUC *and* lag phase from \
the same growth-curve experiment), emit separate rows sharing the same Run_ID.

Rules:
1. paper_id: derive from first/corresponding author surname + year \
(e.g. "Dorado-Morales_2021"). Use the SAME value for every row.
2. experiment_type: use a concise snake_case category — growth_curve_assay, \
evolution_experiment, plasmid_curing_assay, conjugation_assay, MIC_assay, \
plasmid_copy_number_assay, or another descriptive term.
3. run_description: a full sentence summarising the experimental setup, \
including strain, plasmid, key conditions, duration, and confirmation method \
where applicable.
4. field_name: a concise snake_case label for the quantity measured, e.g. \
plasmid_curing_efficiency, area_under_growth_curve, lag_phase_duration, \
plasmid_stability, plasmid_copy_number, plasmid_deletion_detected.
5. raw_value / raw_units: value as reported. For exact numbers use a number. \
For estimates read from figures, prefix with ~ (e.g. ~150). When figure \
extraction data is provided (box-plot medians, line-chart points), use those \
values directly with a ~ prefix since they are read from plots. Only use \
"IN_FIGURE (Fig. X)" if no extraction data exists for that figure.
6. normalized_value / normalized_units: provide when the paper gives or implies \
a normalised form (proportion 0-1, ratio vs control, etc.). Use null if not \
applicable.
7. dispersion_value / dispersion_type: report variability (SD, SEM, IQR, CI) \
when available. For values estimated from box-plots or error bars, note that \
in dispersion_type, e.g. "IQR (estimated from boxplot)".
8. measurement_time_h: timepoint in hours. Use 0 for baseline / day-0, \
convert days to hours (1 day = 24 h). Use null if not time-resolved.
9. source_in_paper: cite the specific table/figure/text location. Use the \
format "Table 2", "Fig. 3A", "Main text p.10". Semicolons for multiple \
sources, e.g. "Fig. 4; main text".
10. medium: include supplements with concentrations where stated, e.g. \
"TSB + chloramphenicol (20 µg/mL) + anhydrotetracycline (100 ng/mL)".
11. mobilization_type: the plasmid's transfer capability — "conjugative", \
"mobilizable", or "non-mobilizable".
12. plasmid_size: in kilobases (number). Use null if not stated.
13. COMPLETENESS IS CRITICAL. You must extract measurements from ALL of \
these sources:
  (a) Every row of every data table (e.g. Table 2 — each strain × plasmid \
combination is a separate measurement row).
  (b) Every figure panel that contains data — box plots, line charts, bar \
charts. Use the FIGURE EXTRACTIONS section for numeric values. Each distinct \
plasmid/strain/condition shown in a figure panel becomes its own row.
  (c) Quantitative results stated in the body text (e.g. "plasmid stability \
was 100% at day 0", "stability decreased to 4.6%").
  (d) Evolution / time-series experiments: emit one row per timepoint per \
condition (e.g. day 0, day 7, day 14, day 21, day 28, day 35 of an \
evolution experiment each become separate rows).
  (e) Supplementary data mentioned in the text (e.g. copy number from Fig. S4).
  A typical paper in this domain yields 50–100 measurement rows. If you have \
fewer than 30, you are likely missing data — re-read the inputs.
14. Look for experimental details in Methods, Results, figure legends, table \
footnotes, and supplementary references.
15. Figure extraction data: when the FIGURE EXTRACTIONS section is present, \
cross-reference group labels (e.g. "pMW2t0 C1", "PFt35 pUR2940t35") with \
strain/plasmid information from Methods text to assign correct run metadata. \
Each box-plot group or line-chart data point that represents a distinct \
measurement should become its own row. The figure crop IDs (e.g. "p008_fig01") \
indicate the PDF page — use the paper's figure numbering (e.g. "Fig. 3A") in \
source_in_paper by matching page numbers and axis labels to figure captions.
16. For box-plot data from figure extractions, use the median as raw_value \
with ~ prefix, the y-axis label to determine raw_units, and report the IQR \
as dispersion_value/dispersion_type. When a figure has per-clone box plots \
(C1, C2, C3), average them into a single representative measurement per \
plasmid condition unless the paper explicitly discusses per-clone differences.

Return exactly this JSON shape:
{{
  "result_type": "combined_extraction",
  "candidates": [
    {{
      "measurement_id": "MEAS_001",
      "run_id": "RUN_001",
      "paper_id": "Author_Year",
      "experiment_type": "string",
      "species": "string or null",
      "strain_id": "string or null",
      "plasmid_name": "string or null",
      "plasmid_family": "string or null",
      "plasmid_size": number or null,
      "mobilization_type": "string or null",
      "medium": "string or null",
      "temperature": number or null,
      "culture_format": "string or null",
      "replicates_biological": integer or null,
      "selection_antibiotic": "string or null",
      "run_description": "full sentence describing experimental setup",
      "field_name": "snake_case quantity name",
      "measurement_time_h": number or null,
      "raw_value": number or "~number" or "IN_FIGURE (Fig. X)" or null,
      "raw_units": "string or null",
      "normalized_value": number or "~number" or null,
      "normalized_units": "string or null",
      "dispersion_value": number or "~number" or null,
      "dispersion_type": "string or null",
      "source_in_paper": "Table/Fig/text reference",
      "confidence": "high | medium | low or null"
    }}
  ],
  "notes": "string or null"
}}"""

COMBINED_JSON_FIX = (
    "\n\nYour previous reply was not valid JSON for the schema. "
    "Return only one corrected JSON object."
)
