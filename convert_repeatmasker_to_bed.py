#!/usr/bin/env python3
"""Convert RepeatMasker .out file to BED format."""

# wget https://hgdownload.gi.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.out.gz
# python convert_repeatmasker_to_bed.py /workspace/caduceus_proj/data/hg38/hg38.fa.out.gz /workspace/caduceus_proj/data/hg38/hg38_repeats.bed


import pandas as pd
import argparse
import gzip

def convert_repeatmasker_to_bed(out_file, bed_file):
    """
    Convert RepeatMasker .out file to BED format.
    
    Args:
        out_file: Path to RepeatMasker .out file (.gz supported)
        bed_file: Output BED file path
    """
    # Read RepeatMasker .out file
    # Skip header lines (start with SW, score, perc, etc.)
    if out_file.endswith('.gz'):
        with gzip.open(out_file, 'rt') as f:
            lines = [line for line in f if not line.startswith(('SW', 'score', 'perc', 'len', 'pos', 'query', 'begin', 'matching', 'repeat', 'repeat', 'pos', 'begin', 'end', 'left', 'no', 'ID', 'left', 'matching', 'repeat', 'class/family', 'repeat', 'begin', 'end', 'left', 'ID', 'class/family'))]
    else:
        with open(out_file, 'r') as f:
            lines = [line for line in f if not line.startswith(('SW', 'score', 'perc', 'len', 'pos', 'query', 'begin', 'matching', 'repeat', 'repeat', 'pos', 'begin', 'end', 'left', 'no', 'ID', 'left', 'matching', 'repeat', 'class/family', 'repeat', 'begin', 'end', 'left', 'ID', 'class/family'))]
    
    # Parse the data
    data = []
    for line in lines:
        if line.strip() and not line.startswith('='):
            parts = line.strip().split()
            if len(parts) >= 15:  # Ensure we have enough columns
                try:
                    score = int(parts[0])
                    perc_div = float(parts[1])
                    perc_del = float(parts[2])
                    perc_ins = float(parts[3])
                    query_seq = parts[4]
                    query_start = int(parts[5])
                    query_end = int(parts[6])
                    query_left = parts[7]
                    strand = parts[8]
                    repeat_name = parts[9]
                    repeat_class = parts[10]
                    repeat_start = int(parts[11])
                    repeat_end = int(parts[12])
                    repeat_left = parts[13]
                    id_num = parts[14]
                    
                    # Convert to BED format (0-based coordinates)
                    data.append({
                        'chr': query_seq,
                        'start': query_start - 1,  # Convert to 0-based
                        'end': query_end,
                        'name': f"{repeat_name}_{repeat_class}",
                        'score': score,
                        'strand': strand
                    })
                except (ValueError, IndexError):
                    continue
    
    # Create DataFrame and save as BED
    df = pd.DataFrame(data)
    df = df.sort_values(['chr', 'start'])
    
    # Save as BED file
    df.to_csv(bed_file, sep='\t', header=False, index=False)
    print(f"Converted {len(df)} repeat regions to BED format: {bed_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert RepeatMasker .out to BED format')
    parser.add_argument('out_file', help='RepeatMasker .out file (.gz supported)')
    parser.add_argument('bed_file', help='Output BED file')
    
    args = parser.parse_args()
    convert_repeatmasker_to_bed(args.out_file, args.bed_file)

if __name__ == "__main__":
    main()
