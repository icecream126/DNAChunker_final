"""Utilities for handling repeat regions in genomic sequences."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch


class RepeatRegionManager:
    """Manages repeat region data and provides utilities for penalization."""
    
    def __init__(self, repeat_bed_file: Optional[str] = None):
        """
        Initialize repeat region manager.
        
        Args:
            repeat_bed_file: Path to BED file containing repeat region coordinates
        """
        self.repeat_regions = {}
        if repeat_bed_file:
            self.load_repeat_regions(repeat_bed_file)
    
    def load_repeat_regions(self, bed_file: str) -> None:
        """
        Load repeat regions from BED file.
        
        BED format: chr, start, end, name, score, strand
        """
        bed_path = Path(bed_file)
        if not bed_path.exists():
            raise FileNotFoundError(f"Repeat BED file not found: {bed_file}")
        
        # Read BED file
        df = pd.read_csv(
            bed_path, 
            sep='\t', 
            names=['chr', 'start', 'end', 'name', 'score', 'strand'],
            dtype={'chr': str, 'start': int, 'end': int}
        )
        
        # Group by chromosome
        for chr_name in df['chr'].unique():
            chr_data = df[df['chr'] == chr_name][['start', 'end', 'name', 'score']].values
            self.repeat_regions[chr_name] = chr_data
    
    def get_repeat_mask(self, chr_name: str, start: int, end: int) -> np.ndarray:
        """
        Get binary mask indicating repeat regions in the given interval.
        
        Args:
            chr_name: Chromosome name
            start: Start position (0-based)
            end: End position (exclusive)
            
        Returns:
            Binary mask where 1 indicates repeat regions, 0 indicates non-repeat
        """
        if chr_name not in self.repeat_regions:
            return np.zeros(end - start, dtype=np.int8)
        
        mask = np.zeros(end - start, dtype=np.int8)
        chr_repeats = self.repeat_regions[chr_name]
        
        for repeat_start, repeat_end, _, _ in chr_repeats:
            # Find overlap between repeat region and query interval
            overlap_start = max(start, repeat_start)
            overlap_end = min(end, repeat_end)
            
            if overlap_start < overlap_end:
                # Mark overlapping positions as repeat regions
                mask[overlap_start - start:overlap_end - start] = 1
        
        return mask
    
    def get_repeat_penalty_weights(self, chr_name: str, start: int, end: int, 
                                 penalty_factor: float = 0.1) -> np.ndarray:
        """
        Get penalty weights for positions in the given interval.
        
        Args:
            chr_name: Chromosome name
            start: Start position (0-based)
            end: End position (exclusive)
            penalty_factor: Penalty factor for repeat regions (0-1, where 1 = full penalty)
            
        Returns:
            Array of weights where repeat regions have lower weights
        """
        repeat_mask = self.get_repeat_mask(chr_name, start, end)
        weights = np.ones(end - start, dtype=np.float32)
        weights[repeat_mask == 1] = penalty_factor
        return weights


def create_repeat_mask_from_sequence(sequence: str, 
                                   min_repeat_length: int = 3,
                                   max_repeat_length: int = 20) -> np.ndarray:
    """
    Create repeat mask by detecting simple repeats in sequence.
    
    Args:
        sequence: DNA sequence
        min_repeat_length: Minimum length of repeat pattern
        max_repeat_length: Maximum length of repeat pattern
        
    Returns:
        Binary mask where 1 indicates repeat regions
    """
    seq_len = len(sequence)
    mask = np.zeros(seq_len, dtype=np.int8)
    
    # Check for tandem repeats of different lengths
    for pattern_len in range(min_repeat_length, min(max_repeat_length + 1, seq_len // 2 + 1)):
        for i in range(seq_len - pattern_len * 2 + 1):
            pattern = sequence[i:i + pattern_len]
            if len(pattern) < pattern_len:
                continue
                
            # Check how many times this pattern repeats
            repeat_count = 1
            j = i + pattern_len
            while j + pattern_len <= seq_len:
                if sequence[j:j + pattern_len] == pattern:
                    repeat_count += 1
                    j += pattern_len
                else:
                    break
            
            # If we found a repeat, mark the positions
            if repeat_count >= 2:
                mask[i:i + pattern_len * repeat_count] = 1
    
    return mask
