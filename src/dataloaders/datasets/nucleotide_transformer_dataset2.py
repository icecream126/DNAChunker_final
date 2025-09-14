"""Nucleotide Transformer Benchmarks Dataset.

From: https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks
"""

import torch
import os
from datasets import load_dataset, load_from_disk
from src.dataloaders.utils.rc import coin_flip, string_reverse_complement


class NucleotideTransformer2Dataset(torch.utils.data.Dataset):

    """
    Loop through fasta file for sequence.
    Returns a generator that retrieves the sequence.
    """

    def __init__(
        self,
        split,
        max_length,
        dataset_name=None,
        d_output=2,  # default binary classification
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        conjoin_train=False,
        conjoin_test=False,
        return_augs=False
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        assert not (conjoin_train and conjoin_test), "conjoin_train and conjoin_test cannot both be True"
        if (conjoin_train or conjoin_test) and rc_aug:
            print("When using conjoin, we turn off rc_aug.")
            rc_aug = False
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

        self.split = split

        # For NT tasks, we use data from InstaDeepAI/nucleotide_transformer_downstream_tasks
        cache_path = '/workspace/huggingface/nt_revised_disk'

        if os.path.exists(cache_path):
            # import pdb; pdb.set_trace()
            self.seqs = load_from_disk(os.path.join(cache_path, split))
            self.seqs = self.seqs.filter(lambda x: x["task"]==dataset_name)
        else:
            # import pdb; pdb.set_trace()
            self.seqs = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
            self.seqs.save_to_disk(cache_path)
            self.seqs = self.seqs[split].filter(lambda x: x["task"]==dataset_name)
        # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = self.seqs[idx]["sequence"]  # only one sequence
        y = self.seqs[idx]["label"]

        if (self.rc_aug or (self.conjoin_test and self.split == "train")) and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(
            x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # append list seems to be faster than append tensor
            seq_ids.append(self.tokenizer.sep_token_id)

        if self.conjoin_train or (self.conjoin_test and self.split != "train"):
            x_rc = string_reverse_complement(x)
            seq_rc = self.tokenizer(
                x_rc,
                add_special_tokens=False,
                padding="max_length" if self.use_padding else None,
                max_length=self.max_length,
                truncation=True,
            )
            seq_rc_ids = seq_rc["input_ids"]  # get input_ids
            # need to handle eos here
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq_rc_ids.append(self.tokenizer.sep_token_id)
            seq_ids = torch.stack((torch.LongTensor(seq_ids), torch.LongTensor(seq_rc_ids)), dim=1)

        else:
            # convert to tensor
            seq_ids = torch.LongTensor(seq_ids)

        # need to wrap in list
        target = torch.LongTensor([y])
        return seq_ids, target
