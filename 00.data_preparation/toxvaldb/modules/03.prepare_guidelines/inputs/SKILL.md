---
name: process-guideline
description: Processes OECD Test Guideline PDFs (Section 4 – Health Effects) for toxicology projects. Given a guideline number, reads the corresponding PDF, extracts structured metadata (species, duration, dose groups, route, observations, haematology, pathology) and study endpoints — from either a structured annex table or the guideline body text — then generates a human-readable Markdown report and machine-readable CSV files. Use this skill whenever the user asks to 1)process an OECD guideline, 2)extract data from an OECD test guideline PDF, 3)generate a report for a guideline (e.g. "process guideline 407", "run process-guideline on TG 408"), 4)update the guideline summary CSV, or 5)retrieve endpoint information from any OECD Section 4 test guideline.
---

# Process OECD Guideline PDF

Process OECD Test Guideline number **$ARGUMENTS** by reading its PDF, extracting a structured summary and study endpoints, and generating report files (markdown + CSV).

## Step 1: Validate and Read the PDF

Use the 'pdf' to access to PDFs.

Read the PDF file at path: `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/inputs/OECD Guidelines/$ARGUMENTS.pdf`

If the file does not exist, inform the user:

> PDF not found at `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/inputs/OECD Guidelines/$ARGUMENTS.pdf`.
> Please download it manually from the OECD website and place it at that path.
> OECD guidelines can be found at: https://www.oecd.org/en/search/publications.html (filter by "OECD Guidelines for the Testing of Chemicals, Section 4")

Then stop processing.

If the file exists, read the ENTIRE PDF to understand its full contents before proceeding to extraction.
**CRITICAL**: If the guideline indicates that it was "deleted" or "cancelled" (e.g. "Test Guideline 401 was deleted"), you MUST IGNORE THIS STATUS. Treat it exactly like an active guideline. Extract all the summary fields, observations, and ALL endpoints from the historical text provided in the PDF just as you would for any other guideline. Do not skip extraction simply because it was deleted.

## Step 2: Extract Summary Fields

Extract the following fields from the guideline body text. OECD test guidelines follow a standard structure with numbered paragraphs and consistent section headings.

- **guideline_number**: The number from the title (e.g., 407, 408)
- **title**: The full title (e.g., "Repeated Dose 28-Day Oral Toxicity Study in Rodents")
- **adoption_date**: The date after "Adopted:" in the header (e.g., "25 June 2025")
- **purpose_scope**: 1-3 sentence summary from the INTRODUCTION and/or INITIAL CONSIDERATIONS sections
- **species**: From the "Selection of animal species" section (e.g., "Rat (preferred); other rodent species may be used")
- **duration**: From the title and PROCEDURE sections (e.g., "28 days", "at least 90 days")
- **animals**: From the "Number and sex of animals" section (e.g., "At least 10 per dose level (5 female + 5 male)")
- **dose_groups**: From the "Dosage" section (e.g., "At least 3 test groups + 1 control group")
- **route**: From the "Administration of doses" section (e.g., "Oral (gavage, diet, or drinking water)")

Also extract key observation details as bullet lists from these sections (sections may be combined or named slightly differently across guidelines):

- **Clinical observations**: From "Observations" section
- **Haematology**: From "Haematology" section (or "Haematology and Clinical Biochemistry")
- **Clinical biochemistry**: From "Clinical biochemistry" section (or combined with haematology)
- **Pathology**: From "Pathology", "Gross necropsy", and "Histopathology" sections

## Step 3: Extract Endpoints

This is the MOST IMPORTANT extraction step. Endpoints are extracted from two possible sources: a structured annex table (Step 3a) or the guideline body text (Step 3b).

### Endpoint Eligibility Test

Before adding any endpoint row, apply this eligibility test:

1. The endpoint MUST be a specific measurable, scorable, countable, weighable, or recordable parameter.
2. The endpoint MUST be parameter-level, not method-level or activity-level.
3. The endpoint MUST belong to OECD Section 4 health-effects guideline content. Do not extract ecotoxicology or non-health-effects endpoints.
4. If a phrase names a procedure, assay model, analysis, test battery, observation activity, broad examination activity, monitoring activity, generic necropsy, generic pathology finding, generic histopathology, or generic organ-weight determination, do NOT extract that phrase as an endpoint. Instead, extract the specific parameters listed under it, if the guideline provides them.
5. A target-specific examination CAN be an endpoint when it names the biological target or observable readout that could support phenotype discovery. For example, extract `iris examination`, `conjunctivae examination`, `fundus examination`, or `refractive media examination`; do not extract broad labels such as `ophthalmological examination`.
6. If a phrase combines multiple independent parameters with "/", "or", or "and", split it into separate endpoint rows when each part is individually measurable. Keep a combined label only when the OECD guideline defines one combined anatomical measurement, such as "prostate and seminal vesicles with coagulating glands".

Reject these method/activity-level labels as endpoints:
- `clinical biochemistry tests`
- `clinical biochemistry measurements`
- `adult intact male assay`
- `female pubertal assay`
- `male pubertal assay`
- `bronchoalveolar lavage fluid analysis`
- `detailed clinical examination`
- `detailed clinical observations`
- `general clinical observation`
- `general clinical observations`
- `ophthalmological examination`
- `clinical observations`
- `gross necropsy`
- `gross necropsy findings`
- `gross pathology findings`
- `haematological tests`
- `haematological determinations`
- `histopathology`
- `full histopathology`
- `organ weight determination`
- `organ weights`
- `morbidity monitoring`
- `mortality monitoring`

Keep or split these concrete parameter-level endpoints:
- Keep `alanine aminotransferase` and `aspartate aminotransferase` as separate endpoints, not "clinical biochemistry tests".
- Keep `erythrocyte count`, `haemoglobin concentration`, and `platelet count` as separate endpoints, not "haematological tests".
- Keep `liver weight`, `kidneys weight`, and `testes weight` as separate endpoints, not "organ weights".
- Keep `liver histopathology`, `kidneys histopathology`, and `testes histopathology` as separate endpoints, not generic "histopathology".
- Keep target-specific ophthalmology endpoints such as `iris examination`, `conjunctivae examination`, `fundus examination`, and `refractive media examination`, but do not keep broad `ophthalmological examination`.
- Keep `sperm motility`, `sperm morphology`, and `sperm count` as separate endpoints.
- Split `total protein or albumin` into `total protein` and `albumin`.
- Split `stillbirths/live births` into `stillbirths` and `live births`.

### Step 3a: Search for Structured Annex Table

Search the PDF for an annex containing endocrine-related endpoints organized in a two-column table.

#### Annex Detection

The annex has VARYING names across different guidelines. Look for ANY annex or appendix whose title contains keywords related to endpoints, endocrine activity, or detection of endocrine disrupters. Known patterns:

- **OECD 407**: "ANNEX 2: RECOMMENDED FOR THE DETECTION OF ENDOCRINE DISRUPTERS (EDS) IN TG 407"
- **OECD 408**: "Annex B. Endpoints recommended for detection of endocrine activity"
- Other guidelines may use: "Annex C", "Appendix", numbered annexes, or different phrasing

The identifying signal is a **two-column table structure** where:
- One column indicates REQUIRED/MANDATORY endpoints
- Another column indicates OPTIONAL endpoints

#### Column Name Variations

You MUST handle these column header variations:
- "Mandatory endpoints" | "Optional endpoints" (as in OECD 407)
- "Required Measures" | "Optional Measures" (as in OECD 408)
- Other possible variations: "Required endpoints", "Recommended endpoints", etc.

Normalize all requirement levels to either **"Required"** or **"Optional"** in the output.

#### Annex Table Structure

The table is organized by CATEGORIES (section headers within the table). Known categories include:
- Weight / Organ Weights
- Histopathology
- Hormones measurement / Serum/Plasma Hormone Analyses
- Serum/Plasma Biochemistry
- Sperm Measures
- Other categories that may appear

For EACH endpoint found in the table, record:
1. **category**: The category/section header it falls under
2. **endpoint_name**: The specific endpoint (e.g., "Testes", "Thyroxine (T4)")
3. **requirement_level**: Normalized to "Required" or "Optional"
4. **source**: Set to "Annex"
5. **footnotes**: Any footnote text associated with this endpoint. Look for superscript numbers or symbols next to endpoint names and capture their corresponding footnote text from the bottom of the annex.

If an annex table is found, set `endpoint_source` to "Annex" and proceed to Step 4.

### Step 3b: Extract Body Text Endpoints (if no annex found)

If no structured annex table was found in Step 3a, extract ALL measurable endpoints from the main guideline body text. Scan every section of the guideline for specific, individually measurable parameters.

#### Categories to Extract

Scan the following sections and extract each specific parameter as a separate endpoint:

| Category | Where to look | Example endpoints |
|---|---|---|
| Clinical Observations | "Observations" section | Body weight, food consumption, water consumption, ophthalmological examination |
| Haematology | "Haematology" section | Erythrocyte count, haemoglobin concentration, platelet count, total leukocyte count, differential leukocyte count, haematocrit, prothrombin time |
| Clinical Biochemistry | "Clinical biochemistry" section | ALT, AST, ALP, GGT, total bilirubin, albumin, total protein, creatinine, urea nitrogen, glucose, calcium, sodium, potassium, cholesterol |
| Urinalysis | "Urinalysis" section or within biochemistry | Volume, specific gravity, pH, protein, glucose, blood/blood cells |
| Organ Weights | "Pathology"/"Necropsy" section (organ weight lists) | Each organ listed separately: liver, kidneys, adrenals, testes, epididymides, ovaries, uterus, thyroid, spleen, brain, heart, pituitary, prostate, seminal vesicles, thymus |
| Histopathology | "Histopathology" section (tissue lists) | Each tissue listed for microscopic examination as a separate endpoint |
| Hormones | Various sections (biochemistry, specialized sections) | T4, T3, TSH, testosterone, LH, FSH, estradiol |
| Reproductive | Reproductive/fertility-specific sections | Sperm motility, sperm morphology, sperm count, oestrous cycle, mating index, fertility index, implantation sites |
| Developmental | Developmental/fetal examination sections | Anogenital distance (AGD), vaginal opening, preputial separation, fetal body weight, skeletal examination, soft tissue examination |

#### Classification Rules

Determine the requirement level from the language used in the guideline:

- **"Required"**: The guideline uses mandatory language — "shall", "must", "required", "are to be", or the parameter is listed explicitly as part of the standard protocol without any qualification
- **"Optional"**: The guideline uses permissive language — "optional", "may", "recommended", "if warranted", "as needed", "if considered relevant", "when appropriate", "if indicated", "additional", "supplementary"

#### Granularity

Extract each parameter as a **separate endpoint**. For example:
- "ALT" and "AST" are two separate endpoints, not "liver enzymes"
- "Testes" and "Ovaries" are two separate organ weight endpoints, not "gonads"
- "Erythrocyte count" and "Haemoglobin concentration" are two separate haematology endpoints
- "Gross necropsy", "clinical observations", "haematological tests", and "organ weights" are section or method labels, not endpoints

#### Output

For EACH endpoint, record:
1. **category**: The category from the table above
2. **endpoint_name**: The specific parameter name
3. **requirement_level**: "Required" or "Optional" (inferred from language)
4. **source**: Set to "Guideline text"
5. **footnotes**: Any qualifying notes from the guideline about how/when to measure this parameter (brief; empty string if none)

Set `endpoint_source` to "Guideline text".

### Step 3c: No Endpoints Found

If neither an annex table (3a) nor body text endpoints (3b) yield any measurable parameters:
- Set `endpoint_source` to "None"
- Set endpoint counts to 0
- Skip the endpoints CSV generation
- Note in the markdown report: "This guideline does not contain extractable endpoints."

## Step 4: Create Output Directory

Run this bash command to ensure the output directory exists:

```bash
mkdir -p 00.data_preparation/toxvaldb/modules/03.prepare_guidelines/outputs/endpoints
```

## Step 5: Generate Markdown Report

Create the file `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/outputs/endpoints/OECD_$ARGUMENTS_report.md` using the Write tool with this structure:

```
# OECD Test Guideline {number}: {title}

**Generated:** {current date YYYY-MM-DD}
**Source PDF:** `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/inputs/OECD Guidelines/{number}.pdf`

## Summary

| Field | Value |
|-------|-------|
| Guideline Number | {number} |
| Title | {title} |
| Adoption Date | {adoption_date} |
| Preferred Species | {species} |
| Study Duration | {duration} |
| Animals per Dose Level | {animals} |
| Dose Groups | {dose_groups} |
| Route of Administration | {route} |

## Purpose and Scope

{purpose_scope - 1-3 sentences}

## Key Observations

### Clinical Observations
{bulleted list of key clinical observation requirements}

### Haematology
{bulleted list of haematological examinations}

### Clinical Biochemistry
{bulleted list of biochemistry determinations}

### Pathology
{bulleted list of pathology/necropsy/histopathology requirements}

## Endpoints

**Source:** {one of: "Annex: {exact annex title from PDF}", "Guideline body text", "None"}
**Column Convention:** {for annex: exact column headers e.g. "Mandatory endpoints | Optional endpoints" | for body text: "Inferred from guideline language" | for none: "N/A"}

{If source is "None": "This guideline does not contain extractable endpoints."}

### Endpoint Summary

| Category | Required | Optional |
|----------|----------|----------|
| {category1} | {count} | {count} |
| {category2} | {count} | {count} |
| ... | ... | ... |
| **Total** | **{total_required}** | **{total_optional}** |

### Detailed Endpoints

| Category | Endpoint | Requirement Level | Source | Footnotes |
|----------|----------|-------------------|--------|-----------|
| {cat} | {endpoint} | Required | Annex | {footnote or empty} |
| {cat} | {endpoint} | Optional | Guideline text | {footnote or empty} |
| ... | ... | ... | ... | ... |

### Footnotes

{If footnotes exist, list them here with their reference numbers. If none, omit this section.}
```

## Step 6: Generate Endpoints CSV

If any endpoints were extracted (from either annex or body text), create `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/outputs/endpoints/OECD_$ARGUMENTS_endpoints.csv` using the Write tool.

Only skip CSV generation if `endpoint_source` is "None" (truly zero endpoints found).

CSV format:
- Header row: `guideline_number,category,endpoint_name,requirement_level,source,footnotes`
- One row per endpoint
- Use proper CSV quoting: wrap any field containing commas in double quotes
- UTF-8 encoding
- `requirement_level` must be exactly "Required" or "Optional"
- `source` must be exactly "Annex" or "Guideline text"
- `footnotes` contains the full footnote text if applicable, empty string otherwise

Example rows:
```
guideline_number,category,endpoint_name,requirement_level,source,footnotes
407,Weight,Testes,Required,Annex,
407,Weight,"Prostate + seminal vesicles with coagulating glands",Required,Annex,
408,Histopathology,Ovaries,Required,Annex,"The condition of the oestrogen-sensitive organs in the female should be assessed with reference to the stage of oestrus cycle at termination"
441,Organ Weights,Ventral prostate (VP),Required,Guideline text,Fresh weight to nearest 0.1 mg (both lobes combined)
441,Hormones,Testosterone (T),Optional,Guideline text,Useful to detect steroid metabolism induction
```

## Step 7: Update Summary CSV

File: `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/outputs/endpoints/OECD_guideline_summary.csv`

**If the file does not exist**, create it with this header:
```
guideline_number,title,adoption_date,species,duration,animals,dose_groups,route,endpoint_source,num_required_endpoints,num_optional_endpoints
```

**If the file already exists**, read it first. If a row for guideline number $ARGUMENTS already exists, remove that old row (update in place). Then append the new row for this guideline.

Write the updated file using the Write tool.

Fields:
- `guideline_number`: integer (e.g., 407)
- `title`: full title (quote if contains commas)
- `adoption_date`: date string as found in PDF
- `species`: species description (quote if contains commas)
- `duration`: duration string
- `animals`: animals description (quote if contains commas)
- `dose_groups`: dose group description (quote if contains commas)
- `route`: route description (quote if contains commas)
- `endpoint_source`: "Annex", "Guideline text", or "None"
- `num_required_endpoints`: integer count
- `num_optional_endpoints`: integer count

## Step 8: Verify and Report

After generating all files, read back the first few lines of each generated file to confirm correct formatting.

Then report to the user:

> **OECD $ARGUMENTS processing complete.**
> - **Title:** {title}
> - **Endpoint source:** {Annex / Guideline text / None}
> - **Endpoints found:** {required_count} required, {optional_count} optional
> - **Files generated:**
>   - `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/outputs/endpoints/OECD_$ARGUMENTS_report.md`
>   - `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/outputs/endpoints/OECD_$ARGUMENTS_endpoints.csv` (if endpoints found)
>   - `00.data_preparation/toxvaldb/modules/03.prepare_guidelines/outputs/endpoints/OECD_guideline_summary.csv` (updated)
