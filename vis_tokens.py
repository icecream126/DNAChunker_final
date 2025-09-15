"""
HNet Token Chunking Visualization Tool

This module provides visualization tools for understanding how HNet's routing module
dynamically chunks tokens based on cosine similarity between adjacent token embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from dataclasses import dataclass


@dataclass
class ChunkingResult:
    """Container for HNet chunking analysis results."""
    input_sequence: str
    token_ids: torch.Tensor
    embeddings: torch.Tensor
    cosine_similarities: torch.Tensor
    boundary_probabilities: torch.Tensor
    boundary_mask: torch.Tensor
    chunks: List[str]
    chunk_lengths: List[int]
    chunk_embeddings: torch.Tensor


class HNetRoutingVisualizer:
    """Visualizer for HNet routing module token chunking."""
    
    def __init__(self, tokenizer, model, device='cpu'):
        """
        Initialize the visualizer.
        
        Args:
            tokenizer: Tokenizer for converting sequences to tokens
            model: HNet model with routing module
            device: Device to run computations on
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.eval()
        
    def analyze_sequence(self, sequence: str, max_length: int = 512) -> ChunkingResult:
        """
        Analyze how HNet chunks a given sequence.
        
        Args:
            sequence: DNA sequence to analyze
            max_length: Maximum sequence length
            
        Returns:
            ChunkingResult with detailed chunking analysis
        """
        # Tokenize sequence
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
            
        tokens = self.tokenizer._tokenize(sequence)
        token_ids = torch.tensor([self.tokenizer._convert_token_to_id(token) for token in tokens])
        
        # Get everything from the model with ret_boundaries=True
        with torch.no_grad():
            hidden_states, all_hidden_states, ratio_loss, boundary_mask = self.model(
                token_ids.unsqueeze(0).to(self.device), 
                ret_boundaries=True
            )
            
            # Get embeddings for visualization
            embeddings = self.model.embeddings(token_ids.unsqueeze(0).to(self.device))
        
        # Calculate cosine similarities for visualization
        with torch.no_grad():
            embeddings_shifted = embeddings[:, 1:, :]  # t
            embeddings_prev = embeddings[:, :-1, :]    # t-1
            
            dot_product = torch.sum(embeddings_shifted * embeddings_prev, dim=-1)
            norm_shifted = torch.linalg.vector_norm(embeddings_shifted, dim=-1)
            norm_prev = torch.linalg.vector_norm(embeddings_prev, dim=-1)
            
            eps = torch.finfo(norm_shifted.dtype).eps
            norm_product = (norm_shifted * norm_prev).clamp(min=eps)
            cosine_similarities = dot_product / norm_product
        
        # Extract chunks
        chunks, chunk_lengths = self._extract_chunks(sequence, boundary_mask[0].cpu())
        
        # Get chunk embeddings (representative tokens at boundaries)
        chunk_embeddings = embeddings[0][boundary_mask[0].bool()].cpu()
        
        return ChunkingResult(
            input_sequence=sequence,
            token_ids=token_ids,
            embeddings=embeddings[0].cpu(),
            cosine_similarities=cosine_similarities[0].cpu(),
            boundary_probabilities=torch.zeros_like(boundary_mask[0].cpu().float()),
            boundary_mask=boundary_mask[0].cpu(),
            chunks=chunks,
            chunk_lengths=chunk_lengths,
            chunk_embeddings=chunk_embeddings
        )
    
    def _extract_chunks(self, sequence: str, boundary_mask: torch.Tensor) -> Tuple[List[str], List[int]]:
        """Extract actual chunk sequences from boundary mask."""
        chunks = []
        chunk_lengths = []
        
        start_idx = 0
        for i, is_boundary in enumerate(boundary_mask):
            if is_boundary and i > 0:  # Don't create empty chunk at start
                chunk = sequence[start_idx:i]
                chunks.append(chunk)
                chunk_lengths.append(len(chunk))
                start_idx = i
        
        # Add final chunk
        if start_idx < len(sequence):
            chunk = sequence[start_idx:]
            chunks.append(chunk)
            chunk_lengths.append(len(chunk))
            
        return chunks, chunk_lengths
    
    def plot_chunking_analysis(self, result: ChunkingResult, figsize=(15, 10)) -> plt.Figure:
        """
        Create comprehensive visualization of HNet chunking.
        
        Args:
            result: ChunkingResult from analyze_sequence
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # 1. Sequence with chunk boundaries
        self._plot_sequence_with_boundaries(axes[0], result)
        
        # 2. Cosine similarities
        self._plot_cosine_similarities(axes[1], result)
        
        # 3. Boundary probabilities
        self._plot_boundary_probabilities(axes[2], result)
        
        # 4. Chunk length distribution
        self._plot_chunk_lengths(axes[3], result)
        
        plt.tight_layout()
        return fig
    
    def _plot_sequence_with_boundaries(self, ax, result: ChunkingResult):
        """Plot DNA sequence with chunk boundaries highlighted."""
        sequence = result.input_sequence
        boundaries = result.boundary_mask.numpy()
        
        # Create color map for different chunks
        chunk_colors = plt.cm.Set3(np.linspace(0, 1, len(result.chunks)))
        
        # Plot sequence
        y_pos = 0.5
        chunk_idx = 0
        current_chunk_start = 0
        
        for i, (char, is_boundary) in enumerate(zip(sequence, boundaries)):
            if is_boundary and i > 0:
                # End current chunk
                ax.text(current_chunk_start + (i - current_chunk_start) / 2, y_pos, 
                       f"Chunk {chunk_idx + 1}", ha='center', va='center', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=chunk_colors[chunk_idx], alpha=0.7))
                chunk_idx += 1
                current_chunk_start = i
            
            # Color the character
            color = chunk_colors[chunk_idx] if chunk_idx < len(chunk_colors) else 'lightgray'
            ax.text(i, y_pos, char, ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.8))
        
        # Add final chunk label
        if current_chunk_start < len(sequence):
            ax.text(current_chunk_start + (len(sequence) - current_chunk_start) / 2, y_pos, 
                   f"Chunk {chunk_idx + 1}", ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=chunk_colors[chunk_idx], alpha=0.7))
        
        ax.set_xlim(-0.5, len(sequence) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title("DNA Sequence with HNet Chunk Boundaries", fontsize=14, fontweight='bold')
        ax.set_xlabel("Position in Sequence")
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    def _plot_cosine_similarities(self, ax, result: ChunkingResult):
        """Plot cosine similarities between adjacent tokens."""
        similarities = result.cosine_similarities.numpy()
        positions = np.arange(1, len(similarities) + 1)
        
        ax.plot(positions, similarities, 'b-', linewidth=2, alpha=0.7, label='Cosine Similarity')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Boundary Threshold')
        
        # Highlight boundary positions
        boundary_positions = np.where(result.boundary_mask.numpy())[0]
        if len(boundary_positions) > 0:
            ax.scatter(boundary_positions[1:], similarities[boundary_positions[1:]-1], 
                      color='red', s=50, zorder=5, label='Actual Boundaries')
        
        ax.set_title("Cosine Similarity Between Adjacent Token Embeddings", fontsize=12)
        ax.set_xlabel("Position (t-1 to t)")
        ax.set_ylabel("Cosine Similarity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_boundary_probabilities(self, ax, result: ChunkingResult):
        """Plot boundary probabilities."""
        probs = result.boundary_probabilities.numpy()
        positions = np.arange(len(probs))
        
        ax.plot(positions, probs, 'g-', linewidth=2, alpha=0.7, label='Boundary Probability')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
        
        # Highlight actual boundaries
        boundary_positions = np.where(result.boundary_mask.numpy())[0]
        if len(boundary_positions) > 0:
            ax.scatter(boundary_positions, probs[boundary_positions], 
                      color='red', s=50, zorder=5, label='Actual Boundaries')
        
        ax.set_title("HNet Boundary Probabilities", fontsize=12)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Boundary Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_chunk_lengths(self, ax, result: ChunkingResult):
        """Plot chunk length distribution."""
        chunk_lengths = result.chunk_lengths
        
        # Bar plot of chunk lengths
        chunk_indices = np.arange(len(chunk_lengths))
        bars = ax.bar(chunk_indices, chunk_lengths, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for i, length in enumerate(chunk_lengths):
            ax.text(i, length + 0.1, str(length), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f"Chunk Length Distribution (Total Chunks: {len(chunk_lengths)})", fontsize=12)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Chunk Length")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_length = np.mean(chunk_lengths)
        std_length = np.std(chunk_lengths)
        ax.axhline(y=mean_length, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_length:.1f}±{std_length:.1f}')
        ax.legend()
    
    def plot_embedding_similarity_heatmap(self, result: ChunkingResult, figsize=(12, 8)) -> plt.Figure:
        """
        Plot heatmap of embedding similarities within and between chunks.
        
        Args:
            result: ChunkingResult from analyze_sequence
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        embeddings = result.embeddings.numpy()
        boundaries = result.boundary_mask.numpy()
        
        # Calculate pairwise cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Full similarity matrix
        im1 = ax1.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title("Token Embedding Similarity Matrix")
        ax1.set_xlabel("Token Position")
        ax1.set_ylabel("Token Position")
        
        # Add boundary lines
        boundary_positions = np.where(boundaries)[0]
        for pos in boundary_positions:
            ax1.axhline(y=pos-0.5, color='black', linewidth=2)
            ax1.axvline(x=pos-0.5, color='black', linewidth=2)
        
        plt.colorbar(im1, ax=ax1)
        
        # Chunk-wise analysis
        chunk_similarities = []
        chunk_labels = []
        
        start_idx = 0
        for i, is_boundary in enumerate(boundaries):
            if is_boundary and i > 0:
                chunk_embeddings = embeddings[start_idx:i]
                if len(chunk_embeddings) > 1:
                    chunk_sim = cosine_similarity(chunk_embeddings)
                    chunk_similarities.append(chunk_sim)
                    chunk_labels.append(f"Chunk {len(chunk_labels) + 1}")
                start_idx = i
        
        # Add final chunk
        if start_idx < len(embeddings):
            chunk_embeddings = embeddings[start_idx:]
            if len(chunk_embeddings) > 1:
                chunk_sim = cosine_similarity(chunk_embeddings)
                chunk_similarities.append(chunk_sim)
                chunk_labels.append(f"Chunk {len(chunk_labels) + 1}")
        
        # Plot chunk similarities
        if chunk_similarities:
            # Create a combined plot for all chunks
            max_chunk_size = max(sim.shape[0] for sim in chunk_similarities)
            combined_sim = np.zeros((len(chunk_similarities), max_chunk_size, max_chunk_size))
            
            for i, sim in enumerate(chunk_similarities):
                combined_sim[i, :sim.shape[0], :sim.shape[1]] = sim
            
            # Average similarity within each chunk
            intra_chunk_sims = [np.mean(sim[np.triu_indices_from(sim, k=1)]) for sim in chunk_similarities]
            
            ax2.bar(range(len(intra_chunk_sims)), intra_chunk_sims, alpha=0.7, color='lightcoral')
            ax2.set_title("Average Intra-Chunk Similarity")
            ax2.set_xlabel("Chunk")
            ax2.set_ylabel("Average Cosine Similarity")
            ax2.set_xticks(range(len(chunk_labels)))
            ax2.set_xticklabels(chunk_labels, rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_chunking_report(self, result: ChunkingResult) -> Dict[str, Any]:
        """
        Generate a comprehensive report of chunking analysis.
        
        Args:
            result: ChunkingResult from analyze_sequence
            
        Returns:
            Dictionary with chunking statistics and analysis
        """
        chunk_lengths = result.chunk_lengths
        cosine_sims = result.cosine_similarities.numpy()
        boundary_probs = result.boundary_probabilities.numpy()
        
        report = {
            'sequence_length': len(result.input_sequence),
            'num_tokens': len(result.token_ids),
            'num_chunks': len(chunk_lengths),
            'compression_ratio': len(result.input_sequence) / len(chunk_lengths),
            'chunk_length_stats': {
                'mean': float(np.mean(chunk_lengths)),
                'std': float(np.std(chunk_lengths)),
                'min': int(np.min(chunk_lengths)),
                'max': int(np.max(chunk_lengths)),
                'median': float(np.median(chunk_lengths))
            },
            'similarity_stats': {
                'mean_cosine_sim': float(np.mean(cosine_sims)),
                'std_cosine_sim': float(np.std(cosine_sims)),
                'min_cosine_sim': float(np.min(cosine_sims)),
                'max_cosine_sim': float(np.max(cosine_sims))
            },
            'boundary_stats': {
                'mean_boundary_prob': float(np.mean(boundary_probs)),
                'num_boundaries': int(np.sum(result.boundary_mask)),
                'boundary_frequency': float(np.sum(result.boundary_mask) / len(result.boundary_mask))
            },
            'chunks': result.chunks,
            'chunk_lengths': chunk_lengths
        }
        
        return report


def create_sample_hnet_model(d_model=128, vocab_size=12):
    """Create a sample HNet model for testing."""
    from hnet_mlm.modeling_hnet import RoutingModule, Downsampler
    
    class SampleHNet(nn.Module):
        def __init__(self, d_model, vocab_size):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, d_model)
            self.routing_module = RoutingModule(d_model)
            self.downsampler = Downsampler()
            
        def forward(self, input_ids):
            embeddings = self.embeddings(input_ids)
            boundary_probs, boundary_mask = self.routing_module(embeddings)
            chunks, chunk_lengths = self.downsampler(embeddings, boundary_mask)
            return embeddings, boundary_probs, boundary_mask, chunks, chunk_lengths
    
    return SampleHNet(d_model, vocab_size)


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    # Create sample model and tokenizer
    model = create_sample_hnet_model()
    from caduceus.tokenization_caduceus import CaduceusTokenizer
    tokenizer = CaduceusTokenizer(model_max_length=512)
    
    # Create visualizer
    visualizer = HNetRoutingVisualizer(tokenizer, model)
    
    # Analyze sequence
    result = visualizer.analyze_sequence(sample_sequence)
    
    # Create visualizations
    fig1 = visualizer.plot_chunking_analysis(result)
    fig2 = visualizer.plot_embedding_similarity_heatmap(result)
    
    # Generate report
    report = visualizer.generate_chunking_report(result)
    print("Chunking Report:")
    print(f"Sequence length: {report['sequence_length']}")
    print(f"Number of chunks: {report['num_chunks']}")
    print(f"Compression ratio: {report['compression_ratio']:.2f}")
    print(f"Average chunk length: {report['chunk_length_stats']['mean']:.2f}")
    
    plt.show()
