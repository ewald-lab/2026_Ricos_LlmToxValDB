import csv
import os
import sys
import argparse

def get_tier(row_data, col_idx, mammal_species, chronic_types, target_units, inhalation_units, excluded_types):
    # Basic filters
    species = row_data[col_idx['SPECIES_COMMON']]
    if species not in mammal_species: return None
    
    study_type = row_data[col_idx['STUDY_TYPE']]
    if study_type == 'acute': return None
    
    toxval_type = row_data[col_idx['TOXVAL_TYPE']]
    if toxval_type in excluded_types: return None
    
    try:
        units = row_data[col_idx['TOXVAL_UNITS']].lower()
        dose_raw = row_data[col_idx['TOXVAL_NUMERIC']]
        dose = float(dose_raw) if dose_raw else 0
        
        is_standard_unit = any(u in units for u in target_units)
        is_inhalation = any(u in units for u in inhalation_units)
        is_chronic = study_type in chronic_types

        if is_standard_unit:
            if dose > 1000: return None
            # Standard Chronic/Subchronic
            if dose <= 500 and is_chronic:
                return 'A'
            return 'B'
        elif is_inhalation:
            # We trust inhalation chronic/subchronic as Tier A if dose is not extreme
            if is_chronic and dose <= 1000:
                return 'A'
            return 'B'
        else:
            # Other units (ml/kg, etc.)
            return 'B'
    except:
        return 'B' 

def main():
    parser = argparse.ArgumentParser(description="Tiered grouping of ToxVal studies by DTXSID")
    parser.add_argument("--input", required=True, help="Input classified CSV file")
    parser.add_argument("--output", required=True, help="Output grouped CSV file")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    # Configuration
    MAMMAL_SPECIES = {
        'Rat', 'Mouse', 'Dog', 'Monkey', 'Rabbit', 'European Rabbit', 
        'Guinea Pig', 'Hamster', 'Pig', 'Primate', 'rodent', 'Dog Other', 'Rhesus', 'Cynomolgus',
        'Cat', 'Cow Family', 'Sheep', 'Goat', 'Ferret', 'Gerbil', 'Mink', 'Mini-pig', 'human',
        'Swine', 'Ovine', 'Caprine'
    }

    EXCLUDED_TYPES = {'LD50', 'LC50', 'ALD', 'LD0', 'LD100', 'LCLo', 'Lethal'}
    CHRONIC_TYPES = {'chronic', 'subchronic'}
    TARGET_UNITS = {'mg/kg-day', 'mg/kg/day', 'mg/kg bw/day', 'mg/kg/d'}
    INHALATION_UNITS = {'mg/m3', 'ppm', 'mg/l', 'ug/l'}

    print(f"Running Tiered Grouping on {input_file}...")

    # dict mapping (DTXSID, effect_idx) -> {'A': count, 'B': count}
    findings = {}

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            col_idx = {col: i for i, col in enumerate(header)}
            
            target_col = "TOXICOLOGICAL_EFFECT_LLM_CLEANED"
            if target_col not in col_idx:
                target_col = [c for c in header if "TOXICOLOGICAL_EFFECT_LLM_CLEANED" in c][0]
            
            start_effect_idx = header.index(target_col) + 1
            effect_cols = header[start_effect_idx:]
            num_effects = len(effect_cols)
            dtxsid_idx = col_idx['DTXSID']
            
            count_total = 0
            count_kept = 0
            
            for row in reader:
                count_total += 1
                dtxsid = row[dtxsid_idx]
                if not dtxsid: continue
                
                tier = get_tier(row, col_idx, MAMMAL_SPECIES, CHRONIC_TYPES, TARGET_UNITS, INHALATION_UNITS, EXCLUDED_TYPES)
                if not tier: continue
                
                count_kept += 1
                
                # Update findings for this compound
                if dtxsid not in findings:
                    findings[dtxsid] = [{'A': 0, 'B': 0} for _ in range(num_effects)]
                
                for i in range(num_effects):
                    if row[start_effect_idx + i] == '1':
                        findings[dtxsid][i][tier] += 1
                
                if count_total % 20000 == 0:
                    print(f"Processed {count_total} rows... (Kept studies: {count_kept})")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nConsolidating findings for {len(findings)} compounds...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write output
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['DTXSID'] + effect_cols)
        for dtxsid, effects_stats in findings.items():
            final_row = [dtxsid]
            for stats in effects_stats:
                # LABEL LOGIC: 1 if at least 1 Tier A OR at least 2 Tier B
                if stats['A'] >= 1 or stats['B'] >= 2:
                    final_row.append(1)
                else:
                    final_row.append(0)
            writer.writerow(final_row)

    print(f"Done. Tiered grouping saved to {output_file}")

if __name__ == "__main__":
    main()
