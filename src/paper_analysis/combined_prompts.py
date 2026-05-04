from __future__ import annotations

COMBINED_SYSTEM = """\
You extract structured experimental data from scientific papers about \
plasmid biology, horizontal gene transfer, and bacterial genetics. \
Produce a denormalized table where every measurement row carries its full \
experimental run metadata. Reply with a single JSON object only — no \
markdown fences, no commentary. Use null for unknown values. \
Aim for MAXIMUM COMPLETENESS: every quantitative result in the paper \
should become its own row."""

COMBINED_USER = """\
You are given three types of input from a scientific paper:
(1) Native PDF tables serialized as markdown.
(2) Body text from the paper (which may include table content from \
scanned/old papers that failed native table detection).
(3) Figure extraction data — structured numeric values already read from \
figure panels by a vision model (box-plot medians/IQR, line-chart points, \
table-image cells). These appear under "--- FIGURE EXTRACTIONS ---".

Your task: produce a DENORMALIZED table where every measurement row carries \
full run metadata. Assign Run_ID (RUN_001, RUN_002, …) per unique \
experiment × strain × plasmid × condition. Assign Measurement_ID (MEAS_001, \
MEAS_002, …) globally and sequentially. When a single run produces multiple \
outcomes (e.g. AUC and lag phase, or transfer frequency at different times), \
emit separate rows sharing the same Run_ID.

CRITICAL: Be MAXIMALLY EXHAUSTIVE. Extract EVERY quantitative value in the \
paper — every parameter, count, percentage, rate, ratio, p-value, \
concentration, duration, temperature, model constant, RMS error, transfer \
rate, growth rate, time point, and experimental design parameter. \
Include setup parameters (temperatures, concentrations, durations, volumes, \
replicate counts, dilution factors) as their own rows. Each distinct numeric \
value from tables, figures, Results, Methods, Discussion, or figure \
captions = one measurement row.

=== CRITICAL: SCANNED/OLD-FORMAT TABLE EXTRACTION ===
Many older papers (pre-2000s) have tables that are NOT detected as native \
PDF tables, so they appear as jumbled text in the body. Look for patterns:
- Column headers followed by rows of numbers (even if spacing is garbled)
- Keywords like "TABLE 2", "TABLE 3", "Table continued" in body text
- Sequences of experiment numbers + numeric values on nearby lines
- Figure captions that contain model parameter values (e.g., "Y2 = 10-9", \
  "Z = 0.999990", "p = 0.16667 h-1")
When you find such text-embedded tables, parse EVERY row. A single table \
with 50+ experiments × 4 parameters = 200+ measurement rows. Do NOT \
summarize — extract each experiment's parameters individually.

=== FIELD RULES ===
- paper_id: first-author surname + year (e.g. "Dorado-Morales_2021"), same \
for every row.
- experiment_type: snake_case. Use one of these canonical types when the \
experiment fits:
  growth_curve_assay, evolution_experiment, plasmid_curing_assay, \
  conjugation_assay, competition_assay, plasmid_stability_assay, \
  plasmid_copy_number_assay, MIC_assay, qPCR_growth_timecourse, \
  qPCR_conjugation_kinetics, transcriptomics_assay, metabolomics_assay, \
  plasmid_transfer_model_assay, biofilm_assay, kill_curve, \
  negative_selection_validation, colony_phenotype_assay, \
  plasmid_persistence_conjugation_loss_fitness, data_availability, \
  conjugation_rate_estimation_assay, plasmid_stability_negative_selection_assay, \
  hospital_epidemiology_and_within_patient_plasmid_transfer, \
  phage_plasmid_conjugation_assay, model_parameter_sensitivity, \
  intestinal_transit_tracer_assay, nonselective_plasmid_persistence_evolution, \
  long_term_plasmid_transfer_model, plasmid_transfer_rate_CF_culture, \
  model_parameter_from_figure_caption.
  For types not listed, create a descriptive snake_case name.
- run_description: full sentence including strain, plasmid, conditions.
- field_name: snake_case quantity. Use these canonical names when applicable:
  * Plasmid dynamics: plasmid_stability, plasmid_loss_frequency, \
    plasmid_loss_frequency_or_rate, plasmid_loss_frequency_single_colony, \
    plasmid_loss_rate_experimental, plasmid_loss_rate_model, \
    plasmid_loss_rate_Luria_Delbruck, plasmid_curing_efficiency, \
    plasmid_copy_number, fraction_plasmid_bearing_cells, \
    host_frequency_plasmid_bearing_cells, \
    plasmid_retained_lines, segregation_factor, segregation_factor_Z, \
    segregants_per_cell_generation, plasmid_deletion_detected, \
    plasmid_size, plasmid_size_or_sequence_feature, duplication_length
  * Transfer rate constants (model parameters): \
    gamma1_donor_to_recipient, gamma2_transconjugant_to_recipient, \
    gamma2_transfer_rate_constant, \
    basic_program_gamma1_donor_to_recipient, \
    basic_program_gamma2_transconjugant_to_recipient, \
    basic_program_gamma1, basic_program_RMS_error, \
    segregation_program_gamma1_donor_to_recipient, \
    segregation_program_gamma2_transconjugant_to_recipient, \
    segregation_program_gamma1, segregation_program_gamma2, \
    segregation_program_RMS_error, \
    shelter_program_gamma1_donor_to_recipient, \
    shelter_program_gamma2_transconjugant_to_recipient, \
    shelter_program_RMS_error, shelter_size, shelter_decline, \
    repressed_fertility_program_gamma1_donor_to_recipient, \
    repressed_fertility_program_gamma2_transconjugant_to_recipient, \
    repressed_fertility_program_RMS_error, \
    repressed_gamma2_after_switch, repressed_fertility_switch_time, \
    RMS_error_ratio_best_remaining_to_repressed
  * Transfer/conjugation: conjugation_rate, conjugation_rate_estimate, \
    conjugation_transfer_frequency, relative_conjugation_efficiency, \
    conjugation_efficiency_density_scaled, conjugation_efficiency_model, \
    conjugation_efficiency_exponential_phase, \
    conjugation_detected_condition, conjugation_detection_status, \
    transconjugants_detected, transconjugants_log10_CFU_mL, \
    proportion_transconjugant_free_cultures_p0, self_transfer_detected, \
    conjugation_interaction_direction, mating_duration
  * Growth/fitness: relative_fitness, area_under_growth_curve, \
    lag_phase_duration, maximum_growth_rate, maximum_OD, \
    carrying_capacity, carrying_capacity_K, growth_temperature, \
    doubling_time, colony_area, colony_morphology
  * Competition: competition_index, selection_coefficient, fitness_cost, \
    fitness_effect_p_value
  * MIC/resistance: MIC_value, antibiotic_concentration, \
    selection_antibiotic_concentration, resistance_genes
  * Structural/genomic: compensatory_mutation_type, \
    compensatory_mutation_locus, plasmid_copy_number_range_min, \
    plasmid_copy_number_range_max
  * Model parameters (general): maximum_conjugation_rate_gammaMAX, \
    dilution_rate, daily_dilution_factor, nutrient_yield_Y, \
    resource_half_saturation_Q, resource_consumption_per_cell_division_e, \
    delay_new_transconjugants_active_donors_lambda_T, \
    delay_exhausted_donors_active_lambdaX, \
    critical_conjugation_efficiency, model_replicates, RMS_error
  * Continuous-flow / gut model: p_flow_rate, gut_volume_r, \
    substrate_concentration_Sr, bacterial_cells_per_g_dry_weight, \
    Ks_n, Ks_n_star, max_growth_rate_mu_n, max_growth_rate_mu_n_star
  * Tracer / transit: fraction_tracer_remaining_start, \
    fraction_tracer_remaining_end, calculated_rate_of_elimination
  * Experimental design: initial_cells_per_well, \
    final_inoculation_density, donor_to_recipient_inoculation_ratio, \
    coculture_volume, number_of_transfers, estimated_generations, \
    experiment_repeats_min, independent_experiments, replicate_count, \
    single_colony_lines_tested, experiment_length, \
    parallel_cocultures_for_LDM, population_bottleneck_small, \
    population_bottleneck_medium, population_bottleneck_large
  * Counts/epidemiology: count, species_count, sequence_type_count, \
    colonized_patients, colonized_patient_fraction, \
    patients_enrolled, total_patients_screened, \
    rectal_swabs_collected, pOXA48_carrying_strains, \
    odds_ratio_between_patient_transfer, \
    odds_ratio_within_patient_transfer, \
    clinical_epidemiology_count_or_rate
  * Transcriptomics/metabolomics: n_de_genes, n_metabolites_detected, \
    n_metabolites_altered_common, rna_reads_plasmid_pct, \
    sos_response_auc
  * qPCR / timecourse: TolC_chromosomal_locus_Ct, \
    TraI_plasmid_locus_Ct, qPCR_sampling_interval, \
    total_cell_number_log10_CFU_mL, lysogens_log10_CFU_mL, \
    new_transconjugants_log10_CFU_mL
  * Statistical tests: p_value, stability_difference_significance, \
    cross_species_LDM_vs_SIM_significance
  For field names not listed above, create a descriptive snake_case name \
  that precisely names the measured quantity.

- raw_value: exact number, "~number" for figure estimates, or \
"IN_FIGURE (Fig. X)" only when no numeric data is available.
- measurement_time_h: timepoint in hours. Convert days to hours \
(day 1 = 24, day 7 = 168, day 14 = 336, day 21 = 504, day 28 = 672, \
day 35 = 840). Use 0 for baseline/time-zero. null if not time-resolved.
- source_in_paper: "Table 2", "Fig. 3A", "Main text p.10", or semicolons \
for multiple: "Fig. 4; main text". IMPORTANT: when the body text explicitly \
states a numeric value AND also cites a figure (e.g. "plasmid was maintained \
at 100% throughout the experiment (Fig. 4)"), use "Fig. 4; main text" — not \
just "Fig. 4". This distinguishes text-stated values from figure-only estimates.
- medium: include supplements with concentrations when reported.
- For box-plot vision data: use median as ~value, IQR as dispersion.
- For log-scale data: record the actual value (not the log), note "log scale" \
in raw_units if relevant.

=== FIGURE EXTRACTION RULES ===
Use the PAGE-TO-FIGURE MAPPING to convert crop IDs (e.g. p008_fig01 = \
page 8) to paper figure numbers (e.g. Fig. 3). Sub-panels (A, B) may share \
a page or span pages — use axis labels and figure captions from the body \
text to determine which sub-panel each crop belongs to.

For per-clone or per-replicate data in box plots: average into one \
representative measurement per condition UNLESS the paper explicitly \
discusses per-clone/per-replicate differences.

=== COMPLETENESS CHECKLIST — extract from ALL of these sources ===

1. TABLES (NATIVE AND SCANNED): Every row of every data table. If Table 1 \
has 15 rows × 3 data columns, that could be 45 measurement rows. Extract \
EVERY cell that contains a quantitative value. Include supplementary tables. \
CRITICAL FOR SCANNED PAPERS: When no native tables are detected, look for \
table content embedded in body text. Older papers often have large tables \
(e.g. Table 2 with 50 experiments, Table 3 with 30 experiments × 4 model \
programs = 120+ rows) that appear as partially garbled text. Parse each \
experiment row individually — do not skip experiments because OCR is messy.

2. FIGURES — TIME-SERIES EXPANSION: For any figure showing measurements \
over time (growth curves, stability curves, population dynamics), extract \
EVERY data point at EVERY timepoint for EVERY condition/strain. Example: \
if a figure tracks host frequency of plasmid-bearing cells for 8 plasmid \
lines across 8 timepoints (0, 100, 200, ..., 700 generations), that is \
8 × 8 = 64 rows from that single figure panel. For a figure with 48 lines \
across 8 timepoints, that is 384 rows. Estimate values from the graph \
using "~value" notation. Do NOT summarize multiple timepoints into one \
row — each (condition × timepoint) pair = one row.

3. FIGURES — MATRICES AND HEATMAPS: For conjugation interaction matrices \
or heatmaps (e.g. showing conjugation rates between N donor × M recipient \
strains), extract EVERY cell. For N=15 strains, that is up to 15 × 15 = \
225 data points. Each cell becomes one row with conjugation_rate (and \
optionally conjugation_detection_status and \
conjugation_interaction_direction rows for the same cell). Read values \
from the color scale or log-scale axis.

4. FIGURES — BOX PLOTS, BAR CHARTS, SCATTER PLOTS: Every data point from \
box plots, bar charts, scatter plots, and survival/stability curves. Use \
vision extraction data when available; estimate from text descriptions \
when vision data is missing.

5. TIME-SERIES DATA (GENERAL): When a figure or table tracks a measurement \
over time (% plasmid-bearing cells over days, CFU counts over hours, \
transfer frequencies at timepoints, bacterial densities), emit ONE ROW \
PER TIMEPOINT PER CONDITION. For N conditions × T timepoints = N×T rows. \
Do NOT collapse time-series into a single summary row.

6. BODY TEXT MEASUREMENTS: Scan ALL Results, Discussion, AND Methods \
paragraphs for quantitative claims: percentages, rates, fold-changes, \
counts, p-values, statistical comparisons, confidence intervals, \
concentrations, volumes, temperatures, durations, ratios. Each distinct \
numeric finding = one row.

7. FIGURE CAPTION PARAMETERS: Figure captions in older papers often \
contain model parameter values (e.g., "Y2 = 10-9 ml/cells-h; \
Z = 0.999990; Y = 0.5 g dry weight per g nutrient; Sr = 4.122 × 10-2 \
g/liter"). Extract EACH parameter as its own row with source = \
"Fig. X caption".

8. CONJUGATION / TRANSFER EXPERIMENTS: For each donor × recipient pair \
and condition, extract: transfer frequency, conjugation rate, \
transconjugant counts, detection status. If measured at multiple time \
points, densities, or with different models (SIM, LDM), emit separate \
rows for each.

9. COMPETITION / FITNESS ASSAYS: For each strain/plasmid combination, \
extract: relative fitness, selection coefficient, competition index, \
growth parameters (AUC, lag, max growth rate, carrying capacity). \
Separate rows for plasmid-bearing vs plasmid-free controls.

10. STABILITY / PERSISTENCE ASSAYS: Track fraction plasmid-bearing over \
time or across transfers. One row per condition per timepoint or transfer. \
If plasmid loss frequencies are measured by different methods (colony \
patching, single colony, Luria-Delbrück), emit separate rows for each.

11. MODEL PARAMETERS AND SIMULATIONS: When the paper fits mathematical \
models (conjugation rate models, population dynamics, epidemiological \
models), extract ALL fitted parameter values for EVERY experiment. \
CRITICAL: If a paper tests 4 model programs (Basic, Segregation, Shelter, \
Repressed Fertility) across 30 experiments, each program produces its own \
set of parameters (gamma1, gamma2, RMS_error, plus program-specific \
params). That is 30 experiments × 4 programs × ~3-5 parameters = 360-600 \
rows. Use program-prefixed field_names: basic_program_gamma1_donor_to_recipient, \
shelter_program_RMS_error, repressed_fertility_switch_time, etc. \
EACH EXPERIMENT × EACH PROGRAM × EACH PARAMETER = ONE ROW.

12. STRUCTURAL / GENOMIC FINDINGS: Deletions, rearrangements, mutations, \
copy number changes, IS element insertions, duplications — emit a row \
for each with appropriate field_name and descriptive raw_value.

13. EXPERIMENTAL DESIGN PARAMETERS: Extract setup values that define \
experimental conditions: antibiotic concentrations, temperatures, \
culture volumes, dilution factors, inoculation densities, mating \
durations, number of replicates, number of transfers, population \
bottleneck sizes, sampling intervals, detection limits.

14. EPIDEMIOLOGICAL / CLINICAL DATA: Patient counts, colonization rates, \
odds ratios, swab counts, strain typing counts, transmission rates, \
within-patient and between-patient transfer statistics.

15. SUPPLEMENTARY DATA: When text references supplementary figures/tables \
(e.g. "copy number did not change — Fig. S4"), emit one row per \
plasmid/strain with raw_value = "IN_FIGURE (Fig. S4)".

=== FINAL CHECK ===
Paper complexity varies widely. Count your rows before returning: \
- A simple paper with limited data: 30-80 rows minimum. \
- A medium paper with several tables and figures: 80-200 rows. \
- A paper with large tables, time-series, or model fitting: 200-500+ rows. \
- A paper with scanned tables of 50+ experiments × multiple model programs \
  can yield 400-600+ rows from tables alone. \
- A paper with high-resolution time-series figures (many conditions × many \
  timepoints) can yield 300-500+ rows from figures alone. \
If your extraction has fewer than 50 rows for a data-rich paper, you are \
almost certainly missing data. Re-read: every table cell → every figure \
data point → every figure caption parameter → every number in Results → \
every parameter in Methods → every model parameter for every experiment → \
every supplementary reference.

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
