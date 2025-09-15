""" 
Nucleotide Transformer V2 Tokenizer wrapper for Hugging Face Transformers.
This wraps the InstaDeepAI/nucleotide-transformer-v2-500m-multi-species tokenizer
while maintaining compatibility with the character tokenizer interface.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers.tokenization_utils import PreTrainedTokenizer


class NTV2Tokenizer(PreTrainedTokenizer):
    def __init__(self, model_name: str = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 
                 model_max_length: int = 512, padding_side: str = 'left', **kwargs):
        """Nucleotide Transformer V2 tokenizer wrapper.
        Args:
            model_name (str): Hugging Face model name for the tokenizer
            model_max_length (int): Model maximum sequence length
            padding_side (str): Padding side ('left' or 'right')
        """
        self.model_name = model_name
        self.model_max_length = model_max_length
        
        # Load the underlying tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        
        # Extract special tokens from the underlying tokenizer
        bos_token = self._tokenizer.bos_token
        eos_token = self._tokenizer.eos_token
        sep_token = self._tokenizer.sep_token
        cls_token = self._tokenizer.cls_token
        pad_token = self._tokenizer.pad_token
        unk_token = self._tokenizer.unk_token
        mask_token = self._tokenizer.mask_token

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def _tokenize(self, text: str) -> List[str]:
        return self._tokenizer.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self._tokenizer.convert_ids_to_tokens(index)

    def convert_tokens_to_string(self, tokens):
        return self._tokenizer.convert_tokens_to_string(tokens)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return self._tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False,
    ) -> List[int]:
        return self._tokenizer.get_special_tokens_mask(
            token_ids_0=token_ids_0,
            token_ids_1=token_ids_1,
            already_has_special_tokens=already_has_special_tokens,
        )

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return self._tokenizer.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """Encode text to token IDs with optional special tokens."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """Decode token IDs back to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def mask_tokens(self, inputs: List[int], mask_token_id: int, special_tokens_mask: Optional[List[int]] = None) -> tuple:
        """Create masked tokens for MLM training.
        Args:
            inputs: Input token IDs
            mask_token_id: ID of the mask token
            special_tokens_mask: Optional mask for special tokens to avoid masking
        Returns:
            tuple: (masked_inputs, labels) where labels are -100 for non-masked tokens
        """
        return self._tokenizer.mask_tokens(inputs, mask_token_id, special_tokens_mask)

    def get_config(self) -> Dict:
        return {
            "model_name": self.model_name,
            "model_max_length": self.model_max_length,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "NTV2Tokenizer":
        cfg = {}
        cfg["model_name"] = config["model_name"]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        # Save the underlying tokenizer
        self._tokenizer.save_pretrained(save_directory, **kwargs)
        
        # Save our config
        cfg_file = Path(save_directory) / "ntv2_tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        # Load our config
        cfg_file = Path(save_directory) / "ntv2_tokenizer_config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                cfg = json.load(f)
            return cls.from_config(cfg)
        else:
            # Fallback to default config
            return cls()
