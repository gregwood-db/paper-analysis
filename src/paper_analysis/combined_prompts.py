from __future__ import annotations

COMBINED_SYSTEM = """\
You extract structured experimental data from scientific papers, producing a \
denormalized table where every measurement row carries its full experimental \
run metadata. Reply with a single JSON object only — no markdown fences, \
no commentary. Use null for unknown values. Aim for MAXIMUM COMPLETENESS: \
every quantitative result in the paper should become its own row."""

COMBINED_USER = """\
You are given three types of input from a scientific paper:
(1) Native PDF tables serialized as markdown.
(2) Body text from the paper.
(3) Figure extraction data — structured numeric values already read from \
figure panels by a vision model (box-plot medians/IQR, line-chart points, \
table-image cells). These appear under "--- FIGURE EXTRACTIONS ---".

Your task: produce a DENORMALIZED table where every measurement row carries \
full run metadata. Assign Run_ID (RUN_001, RUN_002, …) per unique \
experiment × strain × plasmid × condition. Assign Measurement_ID (MEAS_001, \
MEAS_002, …) globally and sequentially. When a single run produces multiple \
outcomes (e.g. AUC and lag phase), emit separate rows sharing the same Run_ID.

=== FIELD RULES ===
- paper_id: first-author surname + year (e.g. "Dorado-Morales_2021"), same \
for every row.
- experiment_type: snake_case (growth_curve_assay, evolution_experiment, \
plasmid_curing_assay, etc.).
- run_description: full sentence including strain, plasmid, conditions.
- field_name: snake_case quantity (plasmid_curing_efficiency, \
area_under_growth_curve, lag_phase_duration, plasmid_stability, \
plasmid_copy_number, plasmid_deletion_detected).
- raw_value: exact number, "~number" for figure estimates, or \
"IN_FIGURE (Fig. X)" only when no numeric data is available.
- measurement_time_h: timepoint in hours (day 7 = 168, day 14 = 336, \
day 21 = 504, day 28 = 672, day 35 = 840). Use 0 for baseline. null if \
not time-resolved.
- source_in_paper: "Table 2", "Fig. 3A", "Main text p.10", or semicolons \
for multiple: "Fig. 4; main text". IMPORTANT: when the body text explicitly \
states a numeric value AND also cites a figure (e.g. "pMW2 was stably \
maintained at 100% throughout the experiment (Fig. 4)"), use "Fig. 4; main \
text" as the source — not just "Fig. 4". This distinguishes text-stated \
baseline/endpoint values from figure-only intermediate estimates.
- medium: include supplements with concentrations.
- For box-plot vision data: use median as ~value, IQR as dispersion.

=== FIGURE EXTRACTION RULES ===
Use the PAGE-TO-FIGURE MAPPING to convert crop IDs (e.g. p008_fig01 = \
page 8) to paper figure numbers (e.g. Fig. 3). Sub-panels (A, B) may share \
a page or span pages — use axis labels and figure captions from the body \
text to determine which sub-panel each crop belongs to.

For per-clone box plots (C1, C2, C3): average into one representative \
measurement per plasmid condition UNLESS the paper explicitly discusses \
per-clone differences.

=== COMPLETENESS CHECKLIST — you MUST extract from ALL of these ===

1. TABLES: Every row of every data table. If Table 2 has 15 strain × \
plasmid rows, emit 15 measurement rows.

2. BOX-PLOT FIGURES: Every box-plot group in FIGURE EXTRACTIONS → one row \
per plasmid/condition with AUC and lag as separate rows. Include ALL figure \
panels across ALL pages (p008 through p013 and beyond).

3. EVOLUTION TIME-SERIES: When a figure tracks a measurement over time \
(e.g. % plasmid-bearing cells over 35 days), emit ONE ROW PER TIMEPOINT \
PER CONDITION. For N plasmids measured at T timepoints, this means N × T \
rows. Even without vision data points, infer values from text (e.g. "stably \
maintained" = ~100 at each timepoint; a declining curve → estimate \
intermediate values). Do NOT collapse into a single summary row.
For source attribution: baseline (day 0) and endpoint values that are \
explicitly stated in the text should use "Fig. X; main text" as source \
with exact numbers. Intermediate timepoints estimated from the figure only \
should use "Fig. X" with ~approximate values.

4. BODY TEXT MEASUREMENTS: Scan ALL Results paragraphs for quantitative \
claims. Capture: stability percentages, plasmid loss events, per-clone \
retransformation results, deletion/rearrangement events (field_name = \
"plasmid_deletion_detected", raw_value = "yes").

5. RETRANSFORMATION EXPERIMENTS: Evolved cured clones re-transformed with \
ancestral or evolved plasmids — each clone × plasmid variant gets its own \
rows. Track C1/C2/C3 and plasmid generation (t0/t35) in strain_id and \
plasmid_name.

6. RETRANSFORMATION STABILITY: Prose descriptions of stability after \
re-introduction → separate rows per clone per variant. For EACH clone \
(C1, C2, C3), emit TWO rows: one for the adapted/evolved plasmid variant \
(typically stable at 100%) and one for the non-adapted variant (e.g. lost \
to 4%). This yields 2 rows × 3 clones = 6 rows for a typical 3-clone \
experiment. Use "Main text p.X" as source.

7. SUPPLEMENTARY DATA: When text references supplementary figures (e.g. \
"copy number did not change — Fig. S4"), emit one row per plasmid/strain \
studied with raw_value = "IN_FIGURE (Fig. S4)" and field_name matching the \
measurement type (e.g. plasmid_copy_number). Use source = "Fig. S4 \
(supplementary); main text p.X" where X is the page referencing it.

8. PLASMID STRUCTURAL CHANGES: When text describes a deletion, \
rearrangement, or structural variant (e.g. "12.8-kb deletion mediated by \
IS elements"), emit a row with field_name = "plasmid_deletion_detected", \
raw_value = "yes". Use source = "Main text p.X; Fig. Y" citing both the \
text and any figure showing the structural comparison.

=== FINAL CHECK ===
A typical paper with tables, evolution assays, growth curves, \
retransformation, and supplementary data yields 65–100 measurement rows. \
If you have fewer than 50 rows, re-read the inputs systematically: \
Tables → Figure captions → Results text → Supplementary references → \
Figure extraction data.

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
