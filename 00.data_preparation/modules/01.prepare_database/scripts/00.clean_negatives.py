import pandas as pd
import re
import numpy as np
import os

import argparse

# --- 1. Parse Arguments ---
parser = argparse.ArgumentParser(description="Clean ToxValDB database (keeping negatives).")
parser.add_argument("--input", required=True, help="Path to the unified input database (TSV).")
parser.add_argument("--output", required=True, help="Path to the cleaned output database (TSV).")
args = parser.parse_args()

input_file = args.input
output_file = args.output

# --- 2. Load Data ---
df = pd.read_csv(input_file, sep='\t', low_memory=False)

# --- 3. Create OECD Expected Guideline ---
def get_guideline_matrix_v3(row):
    st_type = str(row.get('STUDY_TYPE', '')).lower()
    route = str(row.get('EXPOSURE_ROUTE', '')).lower()
    species = str(row.get('SPECIES_COMMON', '')).lower()
    duration_val = str(row.get('STUDY_DURATION_VALUE', ''))
    duration_class = str(row.get('STUDY_DURATION_CLASS', '')).lower()
    generation = str(row.get('GENERATION', ''))
    toxval_type = str(row.get('TOXVAL_TYPE', '')).lower()
    year = row.get('YEAR', 0)

    if 'human' in species: return None

    is_rabbit = 'rabbit' in species
    is_guinea = 'guinea' in species
    is_mouse  = 'mouse' in species
    is_rat    = 'rat' in species
    is_dog    = 'dog' in species
    is_monkey = 'monkey' in species or 'macaque' in species or 'primate' in species
    is_fish   = any(x in species for x in ['danio', 'fish', 'trout', 'carp', 'minnow'])
    is_bird   = any(x in species for x in ['quail', 'chicken', 'duck', 'mallard', 'bird'])

    if is_fish: return "OECD 203"
    if is_bird: return "OECD 205"

    if 'uterotrophic' in st_type: return "OECD 440"
    if 'hershberger' in st_type: return "OECD 441"

    if is_guinea: return "OECD 406"
    if is_rabbit and ('skin' in st_type or 'dermal' in route or 'irrit' in st_type): return "OECD 404"
    if is_rabbit and 'eye' in st_type: return "OECD 405"

    if 'acute' in st_type or 'ld50' in toxval_type:
        if 'oral' in route and (is_rat or is_mouse):
            try:
                y_val = float(str(year).replace(',', '').split('.')[0])
                if y_val >= 2002: return "OECD 423"
                else: return "OECD 401"
            except: return "OECD 401"
        if 'dermal' in route and (is_rat or is_rabbit): return "OECD 402"
        if 'inhal' in route and is_rat: return "OECD 403"

    if ('F1' in generation or 'F2' in generation or 'P0' in generation) and (is_rat or is_mouse):
        return "OECD 416"
    if ('developmental' in st_type or 'fetus' in generation) and (is_rat or is_rabbit or is_mouse):
        return "OECD 414"
    if 'reproduction' in st_type and (is_rat or is_mouse):
         if 'one' in st_type: return "OECD 415"
         return "OECD 421"

    if 'subchronic' in st_type or 'repeat' in st_type or 'short-term' in st_type:
        if 'oral' in route and (is_rat or is_mouse):
            if '28' in duration_val or 'short' in duration_class: return "OECD 407"
            if '90' in duration_val or 'subchronic' in duration_class: return "OECD 408"
            return "OECD 408"
        if 'oral' in route and (is_dog or is_monkey):
            return "OECD 409"
        if 'dermal' in route and (is_rat or is_rabbit or is_guinea): return "OECD 410"
        if 'inhal' in route and is_rat: return "OECD 412"

    if 'chronic' in st_type and (is_rat or is_mouse or is_dog or is_monkey):
        return "OECD 452"

    return None

df['MATRIX_GUIDELINE'] = df.apply(get_guideline_matrix_v3, axis=1)

# --- 4. Initial Data Cleaning ---
df = df[~df["TOXICOLOGICAL_EFFECT"].str.contains("-", na=False)]
df = df.drop(columns=[col for col in df.columns if "_ORIGINAL" in col])
df = df.drop(columns=[
    'QC_CATEGORY', 'QC_STATUS', 'SOURCE_HASH', 'SOURCE_URL', 
    'SUBSOURCE_URL', 'source_filename'
])

# --- 5. Clean Toxicological Effect Column ---
df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].fillna('').astype(str)
df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].str.replace(r';', '|', regex=True)
df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].str.split('|').apply(lambda x: list(dict.fromkeys([i.strip() for i in x if i.strip()])) if isinstance(x, list) else x)
df = df.explode('TOXICOLOGICAL_EFFECT').reset_index(drop=True)
df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].str.lower().str.strip()

# Regex cleaning
df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].str.replace(r'[~><=+\/-]*\s?\d+[\d\.,]*\s?%', '', regex=True)
units_regex = r'(mg\/kg(?:\/day)?|ppm|mg\/m3|mg\/l|g\/kg|µg\/kg|mmol\/l|mg\/dl|mg\/kg\s?bw)'
df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].str.replace(r'(?:at|@)?\s?[~><=]*\s?\d+[\d\.,-]*\s?' + units_regex + r'\b', '', regex=True)
df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].str.replace(r'\([^)]*[\d%][^)]*\)', '', regex=True)
df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Semantic noise filtering (MODIFIED: Negatives are NOT eliminated)
STOP_PHRASES = {
    "no abnormal phenotype detected", "no significant changes", "no treatment-related effects",
    "no adverse effects", "no effects", "no observation", "no changes", "normal",
    "unremarkable", "negative", "none", "na", "no abnormal findings",
    "terminal sacrifice", "scheduled sacrifice", "study termination", "necropsy", "autopsy"
}
# Procedural noise we still want to filter
PROCEDURAL_NOISE = {
    "terminal sacrifice", "scheduled sacrifice", "study termination", "necropsy", "autopsy", "na"
}
df = df[~df['TOXICOLOGICAL_EFFECT'].isin(PROCEDURAL_NOISE)]

# NEGATIVE_PATTERN = r'^no\s+(?:significant|abnormal|treatment|adverse|observed|toxic|pathological|compound)'
# df = df[~df['TOXICOLOGICAL_EFFECT'].str.contains(NEGATIVE_PATTERN, case=False, na=False, regex=True)]

df = df[df['TOXICOLOGICAL_EFFECT'].str.len() >= 2] # Reduced from 4 to 2 to keep "no" or similar if present

# Sex/Species cleanup
def remove_in_sex_species(text):
    pattern_in = r'\b(?:in|/)\s*(?:male?s?|female?s?|mice|rats?|rabbits?|dogs?|pigs?|monkeys?)(?:\s+(?:and|or)\s+(?:in\s+|/)?(?:male?s?|female?s?|mice|rats?|rabbits?|dogs?|pigs?|monkeys?))*'
    pattern_parens = r'\(\s*(?:male?s?|female?s?)(?:\s+(?:and|or)\s+(?:male?s?|female?s?))?(?:\s+only)?\s*\)'
    combined = f'({pattern_in}|{pattern_parens})'
    cleaned = re.sub(combined, '', text, flags=re.IGNORECASE | re.VERBOSE)
    return re.sub(r'\s+', ' ', cleaned).strip(' ,')

df['TOXICOLOGICAL_EFFECT'] = df['TOXICOLOGICAL_EFFECT'].apply(remove_in_sex_species)
df = df.drop_duplicates().reset_index(drop=True)

# --- 6. Biological and Physical Outlier Detection (EDA) ---
lifespan_map = {
    'Rat': 4, 'Mouse': 4, 'Rabbit': 12, 'European Rabbit': 12, 'Human': 120,
    'Dog': 18, 'Guinea Pig': 8, 'Monkey': 40, 'Rhesus Macaque': 40, 'Cat': 25,
    'Hamster': 4, 'Pig': 20, 'Mink': 10, 'American Mink': 10, 'Gerbil': 5,
    'Japanese Quail': 3, 'Chicken': 8, 'Sheep': 12, 'Zebra Danio': 5,
    'Northern Bobwhite Quail': 8, 'Cow Family': 20
}
unit_to_years = {
    'years': 1.0, 'year': 1.0, 'months': 1/12, 'month': 1/12, 'weeks': 1/52.17,
    'week': 1/52.17, 'days': 1/365.25, 'day': 1/365.25, 'hours': 1/(24*365.25),
    'minutes': 1/(60*24*365.25), 'generation': 1.0
}

if 'STUDY_DURATION_VALUE' in df.columns and 'STUDY_DURATION_UNITS' in df.columns:
    df['STUDY_DURATION_YEARS'] = df.apply(
        lambda row: row['STUDY_DURATION_VALUE'] * unit_to_years.get(str(row['STUDY_DURATION_UNITS']).lower(), 0) 
        if pd.notnull(row['STUDY_DURATION_VALUE']) else np.nan, axis=1
    )

mass_conv_sorted = [
    ('mg/kg-day', 1.0), ('mg/kg/day', 1.0), ('ug/kg-day', 0.001), ('ng/kg-day', 0.000001),
    ('mg/kg bw', 1.0), ('ug/kg bw', 0.001), ('ug/kg', 0.001), ('ng/kg', 0.000001),
    ('mg/kg', 1.0), ('g/kg', 1000.0), ('mg/g', 1000.0), ('ug/mg', 1.0)
]

def classify_toxval_unified(row):
    val = row['TOXVAL_NUMERIC']
    unit = str(row['TOXVAL_UNITS']).lower().strip()
    tox_type = str(row['TOXVAL_TYPE']).lower().strip()
    if pd.isnull(val): return None
    if '-1' in unit or '^-1' in unit or 'cancer' in tox_type or 'slope' in tox_type or 'mg/m3' in unit: 
        return None
    if 'ppm' in unit and val > 1000000: return 'Error'
    if unit in ['%', '% v/v', '% diet', '% w/v', '% weight'] and val > 100: return 'Error'
    for u_str, factor in mass_conv_sorted:
        if u_str in unit:
            if val * factor > 1000000: return 'Error'
            break
    if ('mg/l' in unit or 'ppm' in unit) and val > 100000: return 'Error'
    return None

df['TOXVAL_OUTLIER_FLAG'] = df.apply(classify_toxval_unified, axis=1)

# --- 7. Final Filtering ---
bio_outliers_mask = (df['STUDY_DURATION_YEARS'] > df['SPECIES_COMMON'].map(lifespan_map))
plants_mask = (df['SPECIES_SUPERCATEGORY'] == "Flowers, Trees, Shrubs, Ferns") | (df['SPECIES_COMMON'].str.contains('Zebra Danio', case=False, na=False))
suspicious_mask = df['TOXVAL_OUTLIER_FLAG'].notnull()
non_experimental_mask = (df['EXPERIMENTAL_RECORD'] != "experimental")
human_mask = df['SPECIES_COMMON'].str.contains("Human", na=False)
to_eliminate = ["Sheep", "Rhesus Macaque", "Mink", "Gerbil"]
eliminate_mask = df['SPECIES_COMMON'].isin(to_eliminate)

to_remove_mask = bio_outliers_mask | plants_mask | suspicious_mask | non_experimental_mask | human_mask | eliminate_mask
df_cleaned = df[~to_remove_mask].copy()
df_cleaned = df_cleaned.drop(columns=['EXPERIMENTAL_RECORD', 'TOXVAL_OUTLIER_FLAG'])

species_map = {
    "Western European House Mouse": "Mouse",
    "House Mouse": "Mouse",
    "Mouse, Rat": "Rat",
    "Guinea Pig, Rat": "Rat"
}
df_cleaned['SPECIES_COMMON'] = df_cleaned['SPECIES_COMMON'].replace(species_map)

# --- 8. Export ---
df_cleaned.to_csv(output_file, sep='\t', index=False)
