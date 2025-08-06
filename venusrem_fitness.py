import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO
from typing import List, Optional, Union
from tqdm import tqdm

class VenusREMFitness:
    def __init__(
        self,
        model_name: str = "AI4Protein/ProSST-2048",
        device: Optional[str] = None,
        alpha: float = 0.8,
        logit_mode: str = "aa_seq_aln",
        aa_seq_aln_file: Optional[str] = None,
        struc_seq_aln_file: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.logit_mode = logit_mode
        self.aa_seq_aln_file = aa_seq_aln_file
        self.struc_seq_aln_file = struc_seq_aln_file
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    @staticmethod
    def read_seq(fasta: str) -> str:
        for record in SeqIO.parse(fasta, "fasta"):
            return str(record.seq)

    @staticmethod
    def read_multi_fasta(file_path):
        sequences = {}
        current_sequence = ''
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_sequence:
                        sequences[header] = current_sequence.upper().replace('-', '<pad>').replace('.', '<pad>')
                        current_sequence = ''
                    header = line
                else:
                    current_sequence += line
            if current_sequence:
                sequences[header] = current_sequence
        return sequences

    def count_matrix_from_residue_alignment(self, alignment_dict):
        alignment_seqs = list(alignment_dict.values())
        try:
            aln_start, aln_end = list(alignment_dict.keys())[0].split('/')[-1].split('-')
        except:
            aln_start, aln_end = 1, len(alignment_seqs[0])
        tokenized_results = self.tokenizer(alignment_seqs, return_tensors="pt", padding=True)
        alignment_ids = tokenized_results["input_ids"][:,1:-1]
        return alignment_ids, int(aln_start)-1, int(aln_end)

    def count_matrix_from_structure_alignment(self, alignment_dict):
        alignment_seqs = list(alignment_dict.values())
        if len(alignment_seqs) == 0:
            return None
        tokenized_results = self.tokenizer(alignment_seqs, return_tensors="pt", padding=True)
        alignment_ids = tokenized_results["input_ids"][:,1:-1]
        return alignment_ids

    @staticmethod
    def tokenize_structure_sequence(structure_sequence: List[int]) -> torch.Tensor:
        shift_structure_sequence = [i + 3 for i in structure_sequence]
        shift_structure_sequence = [1, *shift_structure_sequence, 2]
        return torch.tensor([shift_structure_sequence], dtype=torch.long)

    def get_hidden_representations(
        self,
        residue_sequence: Union[str, List[str]],
        structure_sequence: Union[str, List[int]],
    ) -> torch.Tensor:
        """
        Returns hidden representations from the VenusREM model.
        """
        if isinstance(residue_sequence, str) and os.path.isfile(residue_sequence):
            sequence = self.read_seq(residue_sequence)
        else:
            sequence = residue_sequence
            
        if isinstance(structure_sequence, str):
            if os.path.isfile(structure_sequence):
                # Read structure sequence from file
                struct_seq_str = self.read_seq(structure_sequence)
                structure_sequence = [int(i) for i in struct_seq_str.split(",")]
            else:
                # Parse comma-separated string
                structure_sequence = [int(i) for i in structure_sequence.split(",")]
        ss_input_ids = self.tokenize_structure_sequence(structure_sequence).to(self.device)
        tokenized_results = self.tokenizer([sequence], return_tensors="pt")
        input_ids = tokenized_results["input_ids"].to(self.device)
        attention_mask = tokenized_results["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=ss_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # Get the last hidden state (excluding special tokens)
            hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            # Remove special tokens (first and last)
            hidden_states = hidden_states[:, 1:-1, :]  # [1, seq_len-2, hidden_dim]
            return hidden_states.squeeze(0)  # [seq_len-2, hidden_dim]

    def get_raw_logits(
        self,
        residue_sequence: Union[str, List[str]],
        structure_sequence: Union[str, List[int]],
    ) -> torch.Tensor:
        """
        Returns raw language model logits (log-softmaxed, no alignment integration).
        """
        if isinstance(residue_sequence, str) and os.path.isfile(residue_sequence):
            sequence = self.read_seq(residue_sequence)
        else:
            sequence = residue_sequence
            
        if isinstance(structure_sequence, str):
            if os.path.isfile(structure_sequence):
                # Read structure sequence from file
                struct_seq_str = self.read_seq(structure_sequence)
                structure_sequence = [int(i) for i in struct_seq_str.split(",")]
            else:
                # Parse comma-separated string
                structure_sequence = [int(i) for i in structure_sequence.split(",")]
        ss_input_ids = self.tokenize_structure_sequence(structure_sequence).to(self.device)
        tokenized_results = self.tokenizer([sequence], return_tensors="pt")
        input_ids = tokenized_results["input_ids"].to(self.device)
        attention_mask = tokenized_results["attention_mask"].to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=ss_input_ids,
            labels=input_ids,
        )
        logits = outputs.logits[0]
        logits = torch.log_softmax(logits[1:-1, :], dim=-1)
        return logits

    def add_alignment_logits(
        self,
        logits: torch.Tensor,
        aa_seq_aln_file: Optional[str] = None,
        struc_seq_aln_file: Optional[str] = None,
        alpha: Optional[float] = None,
        logit_mode: Optional[str] = None,
        sample_size: Optional[int] = None,
        sample_ratio: float = 1.0,
        sample_times: int = 1,
    ) -> torch.Tensor:
        """
        Modifies logits by integrating alignment matrices according to logit_mode and alpha.
        """
        alpha = alpha if alpha is not None else self.alpha
        logit_mode = logit_mode if logit_mode is not None else self.logit_mode
        aa_seq_aln_file = aa_seq_aln_file or self.aa_seq_aln_file
        struc_seq_aln_file = struc_seq_aln_file or self.struc_seq_aln_file
        logits = logits.clone()
        if alpha != 0:
            if logit_mode in ["aa_seq_aln"] and aa_seq_aln_file is not None and struc_seq_aln_file is None:
                alignment_dict = self.read_multi_fasta(aa_seq_aln_file)
                alignment_matrix, aln_start, aln_end = self.count_matrix_from_residue_alignment(alignment_dict)
                for sample in range(sample_times):
                    if sample_ratio < 1.0:
                        sample_size = int(len(alignment_matrix) * sample_ratio)
                        sample_indices = torch.randperm(len(alignment_matrix))[:sample_size]
                        alignment_matrix_sample = alignment_matrix[sample_indices]
                    else:
                        alignment_matrix_sample = alignment_matrix
                    count_matrix = torch.zeros(alignment_matrix_sample.size(1), self.tokenizer.vocab_size)
                    for i in range(alignment_matrix_sample.size(1)):
                        count_matrix[i] = torch.bincount(alignment_matrix_sample[:,i], minlength=self.tokenizer.vocab_size)
                    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(self.device)
                    count_matrix = torch.log_softmax(count_matrix, dim=-1)
                    aln_modify_logits = (1-alpha) * logits[aln_start: aln_end, :] + alpha * count_matrix
                    logits = torch.cat([logits[:aln_start], aln_modify_logits, logits[aln_end:]], dim=0)
            elif logit_mode in ["struc_seq_aln"] and struc_seq_aln_file is not None and aa_seq_aln_file is None:
                alignment_dict = self.read_multi_fasta(struc_seq_aln_file)
                alignment_matrix = self.count_matrix_from_structure_alignment(alignment_dict)
                if alignment_matrix is not None:
                    count_matrix = torch.zeros(alignment_matrix.size(1), self.tokenizer.vocab_size)
                    for i in range(alignment_matrix.size(1)):
                        count_matrix[i] = torch.bincount(alignment_matrix[:,i], minlength=self.tokenizer.vocab_size)
                    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(self.device)
                    count_matrix = torch.log_softmax(count_matrix, dim=-1)
                    logits = (1-alpha) * logits + alpha * count_matrix
            elif (logit_mode in ["aa_seq_aln+struc_seq_aln", "struc_seq_aln+aa_seq_aln"]) and aa_seq_aln_file is not None and struc_seq_aln_file is not None:
                plm_logits = logits.clone()
                alignment_dict = self.read_multi_fasta(struc_seq_aln_file)
                structure_alignment_matrix = self.count_matrix_from_structure_alignment(alignment_dict)
                if structure_alignment_matrix is not None:
                    count_matrix = torch.zeros(structure_alignment_matrix.size(1), self.tokenizer.vocab_size)
                    for i in range(structure_alignment_matrix.size(1)):
                        count_matrix[i] = torch.bincount(structure_alignment_matrix[:,i], minlength=self.tokenizer.vocab_size)
                    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(self.device)
                    count_matrix = torch.log_softmax(count_matrix, dim=-1)
                    logits = (1-alpha) * plm_logits + alpha * count_matrix
                alignment_dict = self.read_multi_fasta(aa_seq_aln_file)
                residue_alignment_matrix, aln_start, aln_end = self.count_matrix_from_residue_alignment(alignment_dict)
                count_matrix = torch.zeros(residue_alignment_matrix.size(1), self.tokenizer.vocab_size)
                for i in range(residue_alignment_matrix.size(1)):
                    count_matrix[i] = torch.bincount(residue_alignment_matrix[:,i], minlength=self.tokenizer.vocab_size)
                count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(self.device)
                count_matrix = torch.log_softmax(count_matrix, dim=-1)
                aln_modify_logits = (1-alpha) * logits[aln_start: aln_end, :] + alpha * count_matrix
                logits = torch.cat([plm_logits[:aln_start], aln_modify_logits, plm_logits[aln_end:]], dim=0)
        return logits

    def get_logits(
        self,
        residue_sequence: Union[str, List[str]],
        structure_sequence: Union[str, List[int]],
        aa_seq_aln_file: Optional[str] = None,
        struc_seq_aln_file: Optional[str] = None,
        alpha: Optional[float] = None,
        logit_mode: Optional[str] = None,
        sample_size: Optional[int] = None,
        sample_ratio: float = 1.0,
        sample_times: int = 1,
    ) -> torch.Tensor:
        """
        Returns logits for the given sequence(s) with alignment integration (convenience method).
        """
        raw_logits = self.get_raw_logits(residue_sequence, structure_sequence)
        return self.add_alignment_logits(
            raw_logits,
            aa_seq_aln_file=aa_seq_aln_file,
            struc_seq_aln_file=struc_seq_aln_file,
            alpha=alpha,
            logit_mode=logit_mode,
            sample_size=sample_size,
            sample_ratio=sample_ratio,
            sample_times=sample_times,
        )

    def predict_fitness(
        self,
        residue_sequence: Union[str, List[str]],
        structure_sequence: Union[str, List[int]],
        mutants: List[str],
        aa_seq_aln_file: Optional[str] = None,
        struc_seq_aln_file: Optional[str] = None,
        alpha: Optional[float] = None,
        logit_mode: Optional[str] = None,
        sample_size: Optional[int] = None,
        sample_ratio: float = 1.0,
        sample_times: int = 1,
    ) -> List[float]:
        """
        Predicts fitness for a list of mutants given a wildtype sequence and structure sequence.
        mutants: list of mutation strings, e.g. ["A23T", "G45D"]
        """
        logits = self.get_logits(
            residue_sequence=residue_sequence,
            structure_sequence=structure_sequence,
            aa_seq_aln_file=aa_seq_aln_file,
            struc_seq_aln_file=struc_seq_aln_file,
            alpha=alpha,
            logit_mode=logit_mode,
            sample_size=sample_size,
            sample_ratio=sample_ratio,
            sample_times=sample_times,
        )
        if isinstance(residue_sequence, str) and os.path.isfile(residue_sequence):
            sequence = self.read_seq(residue_sequence)
        else:
            sequence = residue_sequence
        vocab = self.tokenizer.get_vocab()
        scores = []
        for mutant in mutants:
            pred_score = 0
            for sub_mutant in mutant.split(":"):
                wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
                assert sequence[idx] == wt, f"Wild type mismatch: {sequence[idx]} != {wt}, idx {idx}"
                score = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
                pred_score += score.item()
            scores.append(pred_score)
        return scores
