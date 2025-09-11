"""Character tokenizer for Hugging Face.

"""

from typing import List, Optional, Dict, Sequence, Tuple
from transformers import PreTrainedTokenizer
import torch
import os
import re
import numpy as np

class CaduceusTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]

    def __init__(self,
                 model_max_length: int,
                 characters: Sequence[str] = ("A", "C", "G", "T", "N"),
                 complement_map=None,
                 bos_token="[BOS]",
                 eos_token="[SEP]",
                 sep_token="[SEP]",
                 cls_token="[CLS]",
                 pad_token="[PAD]",
                 mask_token="[MASK]",
                 unk_token="[UNK]",
                 **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Adapted from https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf/blob/main/tokenization_hyena.py
        Args:
            model_max_length (int): Model maximum sequence length.
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following is a list of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            complement_map (Optional[Dict[str, str]]): Dictionary with string complements for each character.
        """
        if complement_map is None:
            complement_map = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
        self.characters = characters
        self.model_max_length = model_max_length

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "left")

        self._complement_map = {}
        for k, v in self._vocab_str_to_int.items():
            complement_id = self._vocab_str_to_int[complement_map[k]] if k in complement_map.keys() else v
            self._complement_map[self._vocab_str_to_int[k]] = complement_id

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    @property
    def complement_map(self) -> Dict[int, int]:
        return self._complement_map

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text.upper())  # Convert all base pairs to uppercase

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)  # Note: this operation has lost info about which base pairs were originally lowercase

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    # Fixed vocabulary with no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        return ()
    
    def _calculate_boundaries(self, text: str, motif_list: Optional[List[str]] = None) -> np.ndarray:
        """
        Calculate boundaries using efficient string operations (DDP-friendly and no file descriptor issues).
        """
        if not text or not motif_list:
            return np.zeros(len(text), dtype=np.int8)
        
        # Filter out empty strings
        valid_motifs = [m for m in motif_list if m]
        if not valid_motifs:
            return np.zeros(len(text), dtype=np.int8)
        
        boundaries = np.zeros(len(text), dtype=np.int8)
        
        # Use efficient string.find() method for each motif
        for motif in valid_motifs:
            if not motif:  # Skip empty motifs
                continue
                
            start = 0
            while True:
                # Find the next occurrence of the motif
                pos = text.find(motif, start)
                if pos == -1:  # No more occurrences
                    break
                
                # Set boundaries at start and end of motif
                boundaries[pos] = 1
                boundaries[pos + len(motif) - 1] = 1
                
                # Move start position to avoid infinite loops
                start = pos + 1
        
        return boundaries

    def __call__(self, text, **kwargs):
        """
        Override the __call__ method to include boundaries and handle padding.
        """
        # This method is designed to work with a single string.
        if not isinstance(text, str):
            raise NotImplementedError(
                "This custom tokenizer __call__ does not support batch encoding."
            )

        max_length = kwargs.get("max_length", self.model_max_length)
        padding = kwargs.get("padding", "do_not_pad")
        truncation = kwargs.get("truncation", False)
        motif_list = kwargs.get("motif_list", None)

        # Truncation
        if truncation and len(text) > max_length:
            # Truncate from the right
            text = text[:max_length]

        # Tokenization
        tokens = self._tokenize(text)
        input_ids = [self._convert_token_to_id(token) for token in tokens]
        
        if motif_list is not None:
            boundaries = self._calculate_boundaries(text, motif_list)
        else:
            boundaries = None

        # Padding
        if padding == "max_length" and max_length is not None:
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                if self.padding_side == "right":
                    input_ids.extend([self.pad_token_id] * pad_length)
                    boundaries = np.pad(boundaries, (0, pad_length), "constant", constant_values=0)
                else:  # 'left'
                    input_ids = ([self.pad_token_id] * pad_length) + input_ids
                    boundaries = np.pad(boundaries, (pad_length, 0), "constant", constant_values=0)

        return {"input_ids": input_ids, "boundaries": boundaries}
