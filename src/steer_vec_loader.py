import torch
import numpy as np
from os import PathLike

BEHAVIORS = [
    "sycophancy", "refusal", "hallucination",
    "power_seeking", "myopia",          # safety
    "verbosity", "formality", "politeness",  # style
    "confidence", "humor", "agreeableness", "evil",  # persona
]

class SteerVecLoader:
    def __init__(self, steer_vec_path: PathLike, layer: int):
        self.steer_vec_path = steer_vec_path
        self.layer = layer
        self.steer_vecs = None
                
    def dim_check(self, vector: torch.Tensor, expected_dim: int):
        if vector.dim() != expected_dim:
            raise ValueError(f"Vector has {vector.dim()} dimensions, expected {expected_dim}")

    def load_steer_vecs(self):
        l = self.layer
        self.steer_vecs = {}
        for b in BEHAVIORS:
            vec_path = f"{self.steer_vec_path}/{b}_layer{l}.pt"
            vec = torch.load(vec_path, map_location=torch.device('cpu'), weights_only=True)
            self.dim_check(vec, 1)
            self.steer_vecs[b] = vec
        return self.steer_vecs
    
    def to_matrix(self):
        missing = [b for b in BEHAVIORS if b not in self.steer_vecs]
        if missing:
            raise KeyError(f"Missing behaviors in loaded file: {missing}")
        if self.steer_vecs is None:
            raise ValueError("Steer vectors not loaded. Call load_steer_vecs() first.")
        # Convert the steer vectors to a matrix form
        return np.stack([self.steer_vecs[b].detach().cpu().float().numpy() for b in BEHAVIORS], axis=0), BEHAVIORS

    def validate(self):
        if self.steer_vecs is None:
            raise ValueError("Steer vectors not loaded. Call load_steer_vecs() first.")
        missing = [b for b in BEHAVIORS if b not in self.steer_vecs]
        if missing:
            raise KeyError(f"Missing behaviors in loaded file: {missing}")
        return np.stack([np.linalg.norm(self.steer_vecs[b].detach().cpu().float().numpy()) for b in BEHAVIORS], axis=0)