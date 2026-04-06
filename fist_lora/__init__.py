"""FiST-LoRA: Fisher-Informed Subspace Training for extremely parameter-efficient LoRA.

The adapted forward pass of every frozen-outer method in this package is

    h = W0 x + s * B (R (A x)),     B in R^{d x r}, A in R^{r x k} frozen, R in R^{r x r} trained,

and the methods differ only in how (B, A, R_init) and the scale ``s`` are chosen
(see :mod:`fist_lora.methods.registry` and ``docs/METHODS.md``).
"""

__version__ = "1.0.0"
