from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CombinedMeasurementCandidate(BaseModel):
    """One measurement row with full run metadata inlined (denormalized)."""

    measurement_id: str = Field(
        ...,
        description="Sequential identifier, e.g. MEAS_001. Globally unique across the paper.",
    )
    run_id: str = Field(
        ...,
        description=(
            "Run identifier, e.g. RUN_001. Multiple measurements from the same "
            "experimental condition share the same Run_ID."
        ),
    )
    paper_id: str = Field(
        ...,
        description=(
            "Identifier derived from first author surname and year, "
            "e.g. 'Smith_2023'. Must be identical for every row."
        ),
    )
    experiment_type: str = Field(
        ...,
        description=(
            "Category: growth_curve_assay, evolution_experiment, "
            "plasmid_curing_assay, conjugation_assay, MIC_assay, "
            "plasmid_copy_number_assay, etc."
        ),
    )
    species: str | None = Field(
        default=None,
        description="Organism species, e.g. 'Staphylococcus aureus'",
    )
    strain_id: str | None = Field(
        default=None,
        description="Strain name / designation, e.g. 'MW2 PFt0', 'RN4220 (lab strain)'",
    )
    plasmid_name: str | None = Field(
        default=None,
        description="Plasmid or construct name, e.g. 'pMW2', 'pUR2940'",
    )
    plasmid_family: str | None = Field(
        default=None,
        description="Plasmid family / replicon type, e.g. 'pSK-family', 'ColE1'",
    )
    plasmid_size: Any = Field(
        default=None,
        description="Plasmid size in kilobases (number, '~2.0', or null)",
    )
    mobilization_type: str | None = Field(
        default=None,
        description="Transfer capability: 'conjugative', 'mobilizable', 'non-mobilizable', etc.",
    )
    medium: str | None = Field(
        default=None,
        description="Growth medium with supplements, e.g. 'TSB + chloramphenicol (20 µg/mL)'",
    )
    temperature: Any = Field(
        default=None,
        description="Incubation temperature in °C (number or null)",
    )
    culture_format: str | None = Field(
        default=None,
        description="Vessel / format, e.g. '96-well microplate (200 µL per well)', 'liquid culture tube (5 mL)'",
    )
    replicates_biological: int | None = Field(
        default=None,
        description="Number of biological replicates",
    )
    selection_antibiotic: str | None = Field(
        default=None,
        description="Antibiotic used for selection, or null if none",
    )
    run_description: str = Field(
        ...,
        description=(
            "Full sentence describing the experimental setup, e.g. "
            "'Plasmid curing efficiency: S. aureus RN4220 carrying pMW2 "
            "transformed with pEMPTY::sgRNA2; cas9 induced with "
            "anhydrotetracycline at 28°C for 16 hours.'"
        ),
    )
    field_name: str = Field(
        ...,
        description=(
            "Measurement quantity name, e.g. 'plasmid_curing_efficiency', "
            "'area_under_growth_curve', 'plasmid_stability'"
        ),
    )
    measurement_time_h: Any = Field(
        default=None,
        description="Timepoint of measurement in hours (number, or null if not time-resolved)",
    )
    raw_value: Any = Field(
        default=None,
        description=(
            "Value as reported. Use number when exact, string with '~' prefix "
            "for estimates from figures, or 'IN_FIGURE (Fig. X)' when only "
            "visible in a plot."
        ),
    )
    raw_units: str | None = Field(
        default=None,
        description="Units as reported, e.g. '%', 'OD595·h', 'copies per chromosome'",
    )
    normalized_value: Any = Field(
        default=None,
        description="Normalized / transformed value, e.g. proportion 0-1, ratio vs control",
    )
    normalized_units: str | None = Field(
        default=None,
        description="Units of normalized value, e.g. 'proportion (0-1)', 'ratio vs MW2 PFt0'",
    )
    dispersion_value: Any = Field(
        default=None,
        description="Variability measure (SD, SEM, IQR, CI bounds), or null",
    )
    dispersion_type: str | None = Field(
        default=None,
        description="Type of dispersion, e.g. 'SD', 'SEM', 'IQR (estimated from boxplot)', '95% CI'",
    )
    source_in_paper: str = Field(
        ...,
        description=(
            "Where this measurement appears: 'Table 2', 'Fig. 3A', "
            "'Main text p.10', 'Fig. 4; main text' (semicolons for multiple sources)"
        ),
    )
    confidence: str | None = Field(
        default=None,
        description="'high' / 'medium' / 'low' or brief caveat",
    )


class CombinedExtractionBatch(BaseModel):
    """LLM output wrapper for combined run+measurement extraction."""

    result_type: Literal["combined_extraction"] = "combined_extraction"
    candidates: list[CombinedMeasurementCandidate] = Field(default_factory=list)
    notes: str | None = Field(
        default=None,
        description="Overall caveats (ambiguous conditions, missing details, etc.)",
    )
