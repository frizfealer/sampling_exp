from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from enum import Enum
import logging
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TorchDevice(Enum):
    cpu = "cpu"
    mps = "mps"
    cuda = "cuda"


class HuggingFaceLLM:
    def check_device(self, device: TorchDevice):
        if device == TorchDevice.mps:
            if not torch.backends.mps.is_available():
                raise ValueError("MPS is not available on this system")
        elif device == TorchDevice.cuda:
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available on this system")

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: TorchDevice = TorchDevice.cpu,
    ):
        self.check_device(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device.value)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device


class CustomSampler:

    def __init__(self, llm: HuggingFaceLLM):
        self.llm = llm

    def compute_entropy(self, probs):
        return -torch.sum(probs * torch.log2(probs), dim=-1)

    def compute_varentropy(self, probs):
        return (
            torch.sum(probs * torch.log2(probs) ** 2, dim=-1)
            - (torch.sum(probs * -torch.log2(probs), dim=-1)) ** 2
        )

    def custom_top_k_sampling(self, logits, k=50, temperature=0.7):
        """
        Custom top-k sampling implementation
        Args:
            logits: Raw logits from model
            k: Number of highest probability tokens to keep
            temperature: Controls randomness (lower = more deterministic)
        """
        # Apply temperature
        logits = logits / temperature

        # Get top k token indices and their probabilities
        top_k_logits, top_k_indices = torch.topk(logits, k)

        # Softmax to get probabilities
        probs = F.softmax(top_k_logits, dim=-1)

        # Sample from the top k tokens
        chosen_idx = torch.multinomial(probs, num_samples=1)

        return (
            top_k_indices[chosen_idx],
            self.compute_entropy(probs),
            self.compute_varentropy(probs),
            probs,
        )

    def custom_nucleus_sampling(self, logits, p=0.9, temperature=0.7):
        """
        Custom nucleus (top-p) sampling implementation
        Args:
            logits: Raw logits from model
            p: Cumulative probability threshold
            temperature: Controls randomness
        """
        # Apply temperature
        logits = logits / temperature

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > p
        # first dim 0: the input tensor dimension
        # second dim: the first index of true
        first_index_of_true = sorted_indices_to_remove.nonzero(as_tuple=True)[0][0]
        if first_index_of_true == 0:
            sorted_indices_to_remove[first_index_of_true] = False
        elif cumulative_probs[first_index_of_true - 1] < p:
            sorted_indices_to_remove[first_index_of_true] = False
        # Set removed tokens to negative infinity
        sorted_logits[sorted_indices_to_remove] = float("-inf")

        # Sample from the filtered distribution
        probs = F.softmax(sorted_logits, dim=-1)
        chosen_idx = torch.multinomial(probs, num_samples=1)

        return (
            sorted_indices[chosen_idx],
            self.compute_entropy(probs + 1e-16),
            self.compute_varentropy(probs + 1e-16),
            probs[~sorted_indices_to_remove],
        )

    def greedy_sampling(self, logits):
        sorted_logits, _ = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        return (
            torch.argmax(logits, dim=-1),
            self.compute_entropy(probs + 1e-16),
            self.compute_varentropy(probs + 1e-16),
            probs[0, :5],
        )

    def generate_text(
        self, prompt, max_length=50, sampling_method="top_k", **sampling_params
    ):
        """
        Generate text using specified sampling method
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            sampling_method: "top_k" or "nucleus"
            sampling_params: Parameters for the sampling method
        """
        device = self.llm.device.value
        # Encode prompt
        input_ids = self.llm.tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Generate tokens one by one
        ans = []
        for _ in tqdm(range(max_length)):
            with torch.no_grad():
                outputs = self.llm.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                # Apply specified sampling method
                if sampling_method == "top_k":
                    next_token, ent, vare, probs = self.custom_top_k_sampling(
                        next_token_logits[0],
                        k=sampling_params.get("k", 50),
                        temperature=sampling_params.get("temperature", 0.7),
                    )
                elif sampling_method == "top_p":  # nucleus sampling
                    next_token, ent, vare, probs = self.custom_nucleus_sampling(
                        next_token_logits[0],
                        p=sampling_params.get("p", 0.9),
                        temperature=sampling_params.get("temperature", 0.7),
                    )
                elif sampling_method == "greedy":
                    next_token, ent, vare, probs = self.greedy_sampling(
                        next_token_logits
                    )
                else:
                    raise ValueError("Invalid sampling method")
                ans.append((next_token, ent.item(), vare.item(), probs.cpu()))
                # Append new token
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                # Check for end of text token
                if next_token.item() == self.llm.tokenizer.eos_token_id:
                    break

        return self.llm.tokenizer.decode(input_ids[0], skip_special_tokens=True), ans
