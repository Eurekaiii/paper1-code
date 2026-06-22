"""
AeroMoDE Configuration & Parameters
=====================================
All tunable parameters for the AeroMoDE expert placement and
inference scheduling pipeline, following Sections III & IV of the paper.

Reference: LaTeX aeromde_main.tex, Sections III (System Model) & IV (Proposed Design).
"""

from dataclasses import dataclass, field


@dataclass
class ChannelConfig:
    """Wireless channel parameters (Section III-B / IV-B)."""
    # Path loss
    beta_0: float = 1.0          # Reference channel gain at 1 m
    alpha: float = 2.5           # Path loss exponent

    # SNR / rate
    N0: float = 1e-12            # Noise power spectral density (W/Hz)
    B: float = 20e6              # Bandwidth (Hz), 20 MHz
    rho_th: float = 0.5          # SNR threshold (linear, -3 dB) for link availability

    # TX power
    P_ground: float = 5.0        # Ground device transmit power (W)


@dataclass
class DeploymentConfig:
    """Weights for the placement scoring function (Section IV-D, Eq. 17)."""
    # BaseScore weights  (all applied AFTER min-max normalization)
    alpha: float = 0.4           # Weight for communication-aware demand A(e,u)
    beta: float = 0.25           # Weight for compute capability B(e,u)
    gamma: float = 0.2           # Weight for network position C(e,u)
    mu: float = 0.15             # Weight for memory penalty W_e

    # Neighbour demand
    eta: float = 0.5             # Remote demand weight in A(e,u) (Eq. 13)

    # Redundancy (Section IV-E, Eq. 14-15)
    nu: float = 0.3              # Redundancy penalty weight
    eta_red: float = 0.5         # Discount for neighbour redundancy vs local


@dataclass
class SchedulingConfig:
    """Inference scheduling parameters (Section IV-E)."""
    lamb: float = 0.1            # Error penalty weight λ (Eq. 23)
                                 # Larger λ → prefer original experts over substitutes


@dataclass
class SimilarityConfig:
    """Expert similarity parameters (Section IV-C)."""
    xi: float = 0.7              # Similarity threshold ξ ∈ [0,1]
                                 # Experts with cosine sim ≥ ξ are substitutable


@dataclass
class SystemConfig:
    """Top-level configuration aggregator."""
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)

    # Deployment constraint
    max_copies_per_expert: int = 1   # R_e^{max}, default single-copy

    # Random seed
    seed: int = 42
