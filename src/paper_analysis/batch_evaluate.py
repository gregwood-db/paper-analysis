"""Evaluate batch extraction results against a multi-paper ground truth spreadsheet.

Compares per-paper candidate_combined.json outputs against the master
RUNS_MEASUREMENTS spreadsheet, computing row-level coverage metrics and
numeric accuracy.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from paper_analysis.config import BatchConfig


@dataclass
class PaperMetrics:
    paper_id: str
    gt_rows: int = 0
    extracted_rows: int = 0
    matched_rows: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    numeric_pairs: int = 0
    mae: float | None = None
    mape: float | None = None
    missing_field_names: list[str] = field(default_factory=list)
    extra_field_names: list[str] = field(default_factory=list)
    field_name_accuracy: float = 0.0


@dataclass
class BatchMetrics:
    papers: list[PaperMetrics] = field(default_factory=list)
    aggregate_precision: float = 0.0
    aggregate_recall: float = 0.0
    aggregate_f1: float = 0.0
    aggregate_mae: float | None = None
    total_gt_rows: int = 0
    total_extracted_rows: int = 0
    total_matched_rows: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "BATCH EVALUATION SUMMARY",
            "=" * 80,
            f"Papers evaluated: {len(self.papers)}",
            f"Total ground truth rows: {self.total_gt_rows}",
            f"Total extracted rows:    {self.total_extracted_rows}",
            f"Total matched rows:      {self.total_matched_rows}",
            f"Aggregate precision:     {self.aggregate_precision:.3f}",
            f"Aggregate recall:        {self.aggregate_recall:.3f}",
            f"Aggregate F1:            {self.aggregate_f1:.3f}",
        ]
        if self.aggregate_mae is not None:
            lines.append(f"Aggregate MAE (numeric): {self.aggregate_mae:.4f}")
        lines.append("")
        lines.append(f"{'Paper_ID':<35} {'GT':>4} {'Ext':>4} {'Match':>5} "
                      f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'MAE':>8}")
        lines.append("-" * 80)
        for p in sorted(self.papers, key=lambda x: x.paper_id):
            mae_s = f"{p.mae:.3f}" if p.mae is not None else "—"
            lines.append(
                f"{p.paper_id:<35} {p.gt_rows:>4} {p.extracted_rows:>4} {p.matched_rows:>5} "
                f"{p.precision:>6.3f} {p.recall:>6.3f} {p.f1:>6.3f} {mae_s:>8}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)

    def gap_report(self) -> str:
        """Actionable per-paper gap analysis."""
        lines = ["\nGAP ANALYSIS (top missing field_names per paper):"]
        for p in sorted(self.papers, key=lambda x: -len(x.missing_field_names)):
            if not p.missing_field_names:
                continue
            top = p.missing_field_names[:10]
            lines.append(f"\n  {p.paper_id} (recall={p.recall:.3f}, missing {len(p.missing_field_names)} field types):")
            for fn in top:
                lines.append(f"    - {fn}")
            if len(p.missing_field_names) > 10:
                lines.append(f"    ... and {len(p.missing_field_names) - 10} more")
        return "\n".join(lines)


def _parse_numeric(val: object) -> float | None:
    """Try to extract a float from a raw value (handles ~prefix, %, etc.)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if math.isnan(val):
            return None
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "nan", "—", "-"):
        return None
    s = s.lstrip("~≈≥≤><")
    s = re.sub(r"[%°]", "", s)
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_field_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace, replace spaces/hyphens with underscore."""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return s


# Common synonym groups: if any name from a group appears in GT or extraction,
# treat them as equivalent.  Built from MASTER_FIELD_LIST + observed mismatches.
_SYNONYM_GROUPS: list[set[str]] = [
    {"plasmid_stability", "fraction_plasmid_bearing_cells", "percentage_plasmid_hosts",
     "plasmid_retention", "host_frequency_plasmid_bearing_cells",
     "plasmid_loss_or_stability_measurement", "relative_plasmid_stability"},
    {"conjugation_rate", "conjugation_rate_estimate", "conjugation_transfer_frequency",
     "transfer_frequency", "conjugation_frequency", "maximum_conjugation_rate_gammamax",
     "maximum_conjugation_rate_gmax"},
    {"relative_conjugation_efficiency", "conjugation_efficiency_density_scaled",
     "conjugation_efficiency_exponential_phase", "conjugation_efficiency_model",
     "conjugation_efficiency_fold_reduction", "conjugation_efficiency_fold_reduction_max",
     "conjugation_efficiency_enhancement_factor"},
    {"carrying_capacity", "carrying_capacity_k"},
    {"maximum_growth_rate", "growth_rate", "max_growth_rate_mu_n",
     "max_growth_rate_mu_n_star", "maximum_growth_rate_mu"},
    {"plasmid_copy_number", "plasmid_copy_number_per_chromosome",
     "plasmid_chromosome_ratio_fplus_population"},
    {"plasmid_loss_frequency", "plasmid_loss_frequency_or_rate",
     "plasmid_loss_rate_experimental", "plasmid_loss_rate_model",
     "plasmid_loss_frequency_colony_patching_day5",
     "plasmid_loss_frequency_naive_day5",
     "plasmid_loss_frequency_single_colony"},
    {"odds_ratio_between_patient_transfer", "between_patient_transmission_odds_ratio"},
    {"odds_ratio_within_patient_transfer", "within_patient_transmission_odds_ratio"},
    {"species_count", "species_observed_in_clinical_outbreak"},
    {"selection_antibiotic_concentration", "antibiotic_concentration",
     "ampicillin_concentration", "chloramphenicol_concentration",
     "kanamycin_concentration", "tetracycline_concentration",
     "streptomycin_concentration"},
    {"segregation_factor", "segregation_factor_z", "segregation_factor_z_model",
     "segregation_factor_human_gut_model"},
    {"mating_duration", "conjugation_duration"},
    {"colony_area", "colony_morphology", "cells_per_mm2", "cells_per_colony"},
    {"compensatory_mutation_type", "compensatory_mutation_locus",
     "compensated_lineages", "chromosomal_mutations_per_lineage"},
    {"resource_consumption_per_cell_division_e", "resource_consumption_per_cell_division_epsilon",
     "resource_consumption_per_division"},
    {"dilution_rate", "daily_dilution_factor"},
    {"relative_fitness", "fitness_cost", "selection_coefficient", "competition_index"},
    {"experiment_repeats_min", "independent_experiments", "replicate_count", "replicates"},
    {"initial_cells_per_well", "initial_density_plating_replicates",
     "final_inoculation_density", "initial_plasmid_bearing_population"},
    {"n_de_genes", "differentially_expressed_genes"},
    {"n_metabolites_detected", "metabolites_detected"},
    {"n_metabolites_altered_common", "metabolites_altered_common"},
    {"rna_reads_plasmid_pct", "plasmid_rna_fraction", "relative_transcription_per_plasmid",
     "relative_transcription_per_plasmid_rep", "relative_transcription_per_plasmid_nptii",
     "relative_transcription_per_plasmid_arac"},
    {"sos_response_auc", "area_under_curve_luminescence_per_od"},
    {"maximum_od", "maximum_od600", "max_od"},
    {"negative_selection_stringency", "ec958_baseline_stringency",
     "selection_system_stringency", "baseline_stringency"},
    {"plasmid_loss_rate_luria_delbruck",
     "true_plasmid_free_colonies_luria_delbruck_confirmed",
     "plasmid_loss_rate"},
    {"gamma_max_best_fit", "gamma_max_lower_sensitivity",
     "gamma_max_upper_sensitivity"},
    {"delay_new_transconjugants_active_lambdat",
     "delay_new_transconjugants_active_donors_lambda_t",
     "lambda_t_dashed_dotted_line", "lambda_t_dashed_line", "lambda_t_solid_line"},
    {"delay_exhausted_donors_active_lambdax",
     "delay_exhausted_donors_return_lambda_x",
     "lambda_x_dashed_dotted_line", "lambda_x_dashed_line", "lambda_x_solid_line"},
    {"donor_to_recipient_inoculation_ratio", "mixing_ratio"},
    {"number_of_transfers", "estimated_generations"},
    {"daily_colonization_probability", "colonized_patient_fraction"},
    # Freter model parameters — gamma transfer rate constants
    {"gamma1_donor_to_recipient", "conjugation_rate_donor_to_recipient",
     "basic_program_gamma1_donor_to_recipient",
     "shelter_program_gamma1_donor_to_recipient",
     "segregation_program_gamma1_donor_to_recipient",
     "repressed_fertility_program_gamma1_donor_to_recipient",
     "basic_program_gamma1", "segregation_program_gamma1"},
    {"gamma2_transconjugant_to_recipient", "conjugation_rate_transconjugant_to_recipient",
     "basic_program_gamma2_transconjugant_to_recipient",
     "shelter_program_gamma2_transconjugant_to_recipient",
     "segregation_program_gamma2_transconjugant_to_recipient",
     "repressed_fertility_program_gamma2_transconjugant_to_recipient",
     "segregation_program_gamma2"},
    {"gamma2_transfer_rate_constant", "transconjugant_transfer_rate_constant"},
    # Freter model RMS errors
    {"basic_program_rms_error", "segregation_program_rms_error",
     "shelter_program_rms_error", "repressed_fertility_program_rms_error",
     "rms_error", "model_rms_error"},
    {"rms_error_ratio_best_remaining_to_repressed", "rms_error_ratio"},
    # Freter model parameters — shelter / repressed
    {"shelter_size", "shelter_population_size"},
    {"shelter_decline", "shelter_decline_rate"},
    {"repressed_gamma2_after_switch", "gamma2_repressed_after_switch"},
    {"repressed_fertility_switch_time", "fertility_switch_time"},
    # Freter tracer / gut parameters
    {"fraction_tracer_remaining_start", "fraction_tracer_remaining_end",
     "tracer_fraction_remaining"},
    {"calculated_rate_of_elimination", "elimination_rate_constant"},
    {"p_flow_rate", "flow_rate_constant", "flow_rate_human_gut_model"},
    {"gut_volume_r", "gut_volume_available_to_e_coli", "gut_dilution_volume"},
    {"ks_n", "saturation_constant_ks_plasmid_free",
     "saturation_constant_ks_plasmid_bearing", "ks_n_star"},
    {"nutrient_yield_y", "nutrient_yield"},
    {"substrate_concentration_sr", "substrate_concentration"},
    {"bacterial_cells_per_g_dry_weight", "cell_density_conversion_factor"},
    {"final_recipients_of_pbr322", "final_pbr322_recipients_k12",
     "final_pbr322_recipients_wildtype"},
    # Area under curve
    {"area_under_growth_curve", "area_under_curve", "auc"},
    # Clinical / epidemiology counts
    {"clinical_epidemiology_count_or_rate", "clinical_count", "patient_count",
     "colonized_patients", "total_patients_screened", "poxA48_carrying_strains",
     "patients_enrolled", "rectal_swabs_collected", "count"},
    # Conjugation detection
    {"conjugation_detection_status", "conjugation_detected", "transfer_detected",
     "conjugation_detected_condition", "transconjugants_detected"},
    {"conjugation_interaction_direction", "interaction_direction",
     "interaction_count", "positive_interaction_count", "negative_interaction_count",
     "interaction_percentage", "positive_interaction_percentage"},
    # Sequence type counts
    {"sequence_type_count", "strain_count"},
    # Plasmid size
    {"plasmid_size", "plasmid_size_or_sequence_feature",
     "plasmid_size_range_min", "plasmid_size_range_max",
     "virulence_plasmid_size_range_min", "virulence_plasmid_size_range_max"},
    # Growth temperature
    {"growth_temperature", "temperature_or_growth_condition", "culture_temperature"},
    # Replicate populations / lines
    {"replicate_populations", "parallel_populations", "plasmid_retained_lines"},
    # Incompatibility / screening counts
    {"incompatibility_group_count", "plasmid_screen_count", "plasmid_screen_inclusion"},
    # Plate measurements
    {"plate_measurement_replicates_min", "plate_measurement_replicates_max",
     "technical_replicates"},
    # Relative growth
    {"relative_growth_alpha", "relative_growth", "growth_alpha"},
    # Conjugation range values
    {"conjugation_transfer_frequency_range_min", "conjugation_transfer_frequency_range_max",
     "transfer_frequency_range"},
    # Self-transfer and curing
    {"self_transfer_detected", "self_transmissible"},
    {"incfii_curing_effect", "curing_effect"},
    # Substrate / temperature effect
    {"substrate_effect_fold_change", "substrate_fold_change"},
    {"temperature_effect_fold_change", "temperature_fold_change"},
    # Gene overlap / intergenic distance
    {"gene_overlap_pslt029_pslt030", "gene_overlap", "gene_overlap_bp"},
    {"intergenic_distance_ccdbst_pslt029", "intergenic_distance_pslt030_rsdb",
     "intergenic_distance", "intergenic_distance_bp"},
    {"segregant_fraction_increase_fold", "segregant_fold_increase", "fold_increase_segregants",
     "fold_increase_segregants_fraction"},
    {"polycistronic_operon_gene_count", "operon_gene_count", "genes_in_operon"},
    {"promoter_region_length", "promoter_length"},
    # Arabinose / induction
    {"arabinose_induction", "arabinose_concentration", "induction_time",
     "growth_induction_time"},
    # Assay parameters
    {"assay_detection_limit", "detection_limit"},
    {"assay_population_size", "population_size"},
    # Significance tests
    {"cross_species_ldm_vs_sim_significance",
     "cross_species_sim_standard_vs_truncated_significance",
     "within_species_sim_truncated_vs_ldm_significance",
     "within_species_vs_cross_species_ldm_significance",
     "stability_difference_significance", "p_value"},
    # Proportion / probability measures
    {"proportion_transconjugant_free_cultures_p0", "p0_fraction"},
    # LDM parallel cocultures
    {"parallel_cocultures_for_ldm", "ldm_cocultures"},
    # Copy number ranges (already covered by plasmid_copy_number group, but add ranges)
    {"plasmid_copy_number_range_min", "plasmid_copy_number_range_max"},
    # Recipient counts
    {"recipient_positive_count", "recipient_count"},
    # No transconjugant detection
    {"no_transconjugant_detection_limit", "transconjugant_detection_limit",
     "transconjugant_detection_limit_volunteer_m", "conjugation_detection_limit"},
    # Duplicated fragments
    {"duplicated_nptii_fragment", "duplicated_oriv_fragment", "duplication_length"},
    # Fragment lengths
    {"inserted_random_fragment_length", "random_fragment_length"},
    # Adsorption / phage scaling
    {"adsorption_to_plasmid_carriers_common_donor_scaling",
     "conjugation_to_lysogen_scaling_independent_donor",
     "conjugation_to_lysogens_common_donor_scaling"},
    # Igler density parameters
    {"common_donor_high_starting_density", "common_donor_low_starting_density"},
    {"common_independent_donor_biological_replicates",
     "common_independent_donor_replicates"},
    # Phage resistance
    {"ancestral_phage_resistance_screen_colonies_per_replicate",
     "phage_resistance_colonies"},
    # Model output / simulation fields (Igler)
    {"model_double_transformants_log10_cfu_ml", "double_transformants_log10_cfu_ml"},
    {"model_lysogens_log10_cfu_ml", "lysogens_log10_cfu_ml"},
    {"model_total_cell_number_log10_cfu_ml", "total_cell_number_log10_cfu_ml"},
    {"model_transconjugants_log10_cfu_ml", "transconjugants_log10_cfu_ml",
     "new_transconjugants_log10_cfu_ml"},
    {"dryad_main_experiments_file_size", "data_file_size"},
    {"independent_donor_transconjugants_vs_single_donor_t",
     "independent_vs_single_donor_transconjugants_t_test",
     "independent_vs_common_donor_t_test"},
    # Segregants / stability fractions (Lobato-Marquez)
    {"segregants_fraction", "fraction_segregants", "segregants_per_cell_generation"},
    # Cell density (Kosterlitz)
    {"cell_density", "cell_density_cfu_ml", "maximum_population_density"},
    # Fitness p-value
    {"fitness_effect_p_value", "fitness_p_value", "selection_coefficient_p_value"},
    # Transfer interval
    {"transfer_interval", "transfer_interval_h", "serial_transfer_interval"},
    # Experiment length and design
    {"experiment_length", "experiment_duration", "experiment_length_h",
     "experiment_length_days"},
    # qPCR Ct values (Wan)
    {"tolc_chromosomal_locus_ct", "chromosomal_locus_ct"},
    {"trai_plasmid_locus_ct", "plasmid_locus_ct"},
    {"qpcr_sampling_interval", "sampling_interval_h"},
    # Resource saturation (Wan model)
    {"resource_half_saturation_q", "half_saturation_constant"},
    # Deletion detection
    {"plasmid_deletion_detected", "deletion_detected"},
    # Single colony confirmation
    {"true_plasmid_free_colonies_single_colony_confirmed",
     "colonies_confirmed_plasmid_lost", "confirmed_plasmid_free_colonies",
     "plasmid_loss_validation"},
]

_SYNONYM_MAP: dict[str, str] = {}
for _group in _SYNONYM_GROUPS:
    canonical = sorted(_group)[0]
    for name in _group:
        _SYNONYM_MAP[name] = canonical


def _canonicalize_field_name(name: str) -> str:
    n = _normalize_field_name(name)
    return _SYNONYM_MAP.get(n, n)


def _match_key(row: dict | pd.Series, is_gt: bool = True) -> tuple:
    """Build a matching key from a row: (paper_id, canonical_field_name, time_h_bucket).

    For ground truth rows (pandas Series), access via column names.
    For extracted rows (dicts), access via dict keys.
    """
    if is_gt:
        paper_id = str(row.get("Paper_ID", "")).strip()
        field_name = _canonicalize_field_name(str(row.get("Field_name", "")))
        time_raw = row.get("Measurement_time_h")
    else:
        paper_id = str(row.get("paper_id", row.get("Paper_ID", ""))).strip()
        field_name = _canonicalize_field_name(str(row.get("field_name", row.get("Field_name", ""))))
        time_raw = row.get("measurement_time_h", row.get("Measurement_time_h"))

    time_h = _parse_numeric(time_raw)
    time_bucket = round(time_h, 1) if time_h is not None else None
    return (paper_id, field_name, time_bucket)


def _match_key_loose(row: dict | pd.Series, is_gt: bool = True) -> tuple:
    """Looser matching key: (paper_id, canonical_field_name) — ignoring time."""
    if is_gt:
        paper_id = str(row.get("Paper_ID", "")).strip()
        field_name = _canonicalize_field_name(str(row.get("Field_name", "")))
    else:
        paper_id = str(row.get("paper_id", row.get("Paper_ID", ""))).strip()
        field_name = _canonicalize_field_name(str(row.get("field_name", row.get("Field_name", ""))))
    return (paper_id, field_name)


def evaluate_paper(
    paper_id: str,
    gt_df: pd.DataFrame,
    extracted: list[dict],
) -> PaperMetrics:
    """Evaluate one paper's extraction against its ground truth subset."""
    gt_paper = gt_df[gt_df["Paper_ID"] == paper_id]
    metrics = PaperMetrics(
        paper_id=paper_id,
        gt_rows=len(gt_paper),
        extracted_rows=len(extracted),
    )

    if not len(gt_paper) or not extracted:
        return metrics

    # Build key sets for matching (strict: field_name + time)
    gt_keys: dict[tuple, list[int]] = {}
    for idx, row in gt_paper.iterrows():
        k = _match_key(row, is_gt=True)
        gt_keys.setdefault(k, []).append(idx)

    ext_keys: dict[tuple, list[int]] = {}
    for i, row in enumerate(extracted):
        k = _match_key(row, is_gt=False)
        ext_keys.setdefault(k, []).append(i)

    # Count matched rows (strict match first, then loose fallback)
    matched_gt_indices: set[int] = set()
    matched_ext_indices: set[int] = set()
    numeric_errors: list[float] = []
    numeric_pct_errors: list[float] = []

    for k, gt_idxs in gt_keys.items():
        if k in ext_keys:
            ext_idxs = ext_keys[k]
            for gi, ei in zip(gt_idxs, ext_idxs):
                matched_gt_indices.add(gi)
                matched_ext_indices.add(ei)

                gt_val = _parse_numeric(gt_paper.loc[gi, "Raw_value"])
                ext_val = _parse_numeric(extracted[ei].get("raw_value", extracted[ei].get("Raw_value")))
                if gt_val is not None and ext_val is not None:
                    err = abs(gt_val - ext_val)
                    numeric_errors.append(err)
                    if abs(gt_val) > 1e-9:
                        numeric_pct_errors.append(err / abs(gt_val))

    # Loose fallback for unmatched ground truth rows
    gt_loose_keys: dict[tuple, list[int]] = {}
    for idx in set(gt_paper.index) - matched_gt_indices:
        row = gt_paper.loc[idx]
        k = _match_key_loose(row, is_gt=True)
        gt_loose_keys.setdefault(k, []).append(idx)

    ext_loose_keys: dict[tuple, list[int]] = {}
    for i in set(range(len(extracted))) - matched_ext_indices:
        row = extracted[i]
        k = _match_key_loose(row, is_gt=False)
        ext_loose_keys.setdefault(k, []).append(i)

    for k, gt_idxs in gt_loose_keys.items():
        if k in ext_loose_keys:
            ext_idxs = ext_loose_keys[k]
            for gi, ei in zip(gt_idxs, ext_idxs):
                if gi not in matched_gt_indices and ei not in matched_ext_indices:
                    matched_gt_indices.add(gi)
                    matched_ext_indices.add(ei)

                    gt_val = _parse_numeric(gt_paper.loc[gi, "Raw_value"])
                    ext_val = _parse_numeric(extracted[ei].get("raw_value", extracted[ei].get("Raw_value")))
                    if gt_val is not None and ext_val is not None:
                        err = abs(gt_val - ext_val)
                        numeric_errors.append(err)
                        if abs(gt_val) > 1e-9:
                            numeric_pct_errors.append(err / abs(gt_val))

    metrics.matched_rows = len(matched_gt_indices)
    metrics.precision = len(matched_ext_indices) / len(extracted) if extracted else 0.0
    metrics.recall = len(matched_gt_indices) / len(gt_paper) if len(gt_paper) else 0.0
    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

    metrics.numeric_pairs = len(numeric_errors)
    if numeric_errors:
        metrics.mae = sum(numeric_errors) / len(numeric_errors)
    if numeric_pct_errors:
        metrics.mape = sum(numeric_pct_errors) / len(numeric_pct_errors)

    # Field-level gap analysis
    gt_fields = set(_canonicalize_field_name(str(v)) for v in gt_paper["Field_name"].dropna().unique())
    ext_fields = set(
        _canonicalize_field_name(str(r.get("field_name", r.get("Field_name", ""))))
        for r in extracted
    )
    metrics.missing_field_names = sorted(gt_fields - ext_fields)
    metrics.extra_field_names = sorted(ext_fields - gt_fields)
    if gt_fields:
        metrics.field_name_accuracy = len(gt_fields & ext_fields) / len(gt_fields)

    return metrics


def _remap_extracted_paper_ids(
    extracted: list[dict], gt_paper_id: str, batch_cfg: BatchConfig
) -> list[dict]:
    """Remap LLM-generated paper_id values to the ground-truth Paper_ID.

    The LLM generates freeform IDs like 'Dorado-Morales_2021' while ground
    truth uses numbered prefixes like '002_DoradoMorales_2021'.  We match
    the extracted rows to the ground-truth paper by overwriting paper_id
    with the canonical gt_paper_id.
    """
    out: list[dict] = []
    for row in extracted:
        row = dict(row)
        row["paper_id"] = gt_paper_id
        row["Paper_ID"] = gt_paper_id
        out.append(row)
    return out


def run_batch_evaluation(batch_cfg: BatchConfig) -> BatchMetrics:
    """Evaluate all papers' combined extractions against the ground truth."""
    gt_path = batch_cfg.ground_truth_path
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    gt_df = pd.read_excel(gt_path, sheet_name=batch_cfg.ground_truth_sheet)
    paper_ids = sorted(gt_df["Paper_ID"].dropna().unique())

    batch_metrics = BatchMetrics()
    total_numeric_errors: list[float] = []

    for paper_id in paper_ids:
        combined_path = (
            batch_cfg.artifacts_dir / paper_id / "text" / "candidate_combined.json"
        )
        extracted: list[dict] = []
        if combined_path.is_file():
            try:
                data = json.loads(combined_path.read_text(encoding="utf-8"))
                extracted = data.get("candidates", [])
            except (json.JSONDecodeError, OSError):
                pass

        extracted = _remap_extracted_paper_ids(extracted, paper_id, batch_cfg)
        pm = evaluate_paper(paper_id, gt_df, extracted)
        batch_metrics.papers.append(pm)
        batch_metrics.total_gt_rows += pm.gt_rows
        batch_metrics.total_extracted_rows += pm.extracted_rows
        batch_metrics.total_matched_rows += pm.matched_rows
        if pm.mae is not None:
            total_numeric_errors.extend(
                [pm.mae] * pm.numeric_pairs
            )

    if batch_metrics.total_extracted_rows:
        batch_metrics.aggregate_precision = (
            batch_metrics.total_matched_rows / batch_metrics.total_extracted_rows
        )
    if batch_metrics.total_gt_rows:
        batch_metrics.aggregate_recall = (
            batch_metrics.total_matched_rows / batch_metrics.total_gt_rows
        )
    p, r = batch_metrics.aggregate_precision, batch_metrics.aggregate_recall
    if p + r > 0:
        batch_metrics.aggregate_f1 = 2 * p * r / (p + r)

    if total_numeric_errors:
        batch_metrics.aggregate_mae = sum(total_numeric_errors) / len(total_numeric_errors)

    return batch_metrics
