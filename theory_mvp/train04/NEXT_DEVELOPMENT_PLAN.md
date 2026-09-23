# Development plan after the fixed 500-update pilot

The completed calibration panels show near-chance deterministic-bit prediction across all three seeds. More sampling repetitions cannot fix that denoiser error. Preserve the primary pilot and its terminal checkpoints as a negative result. The next development task is conditional competence, before another multi-seed campaign or broader language benchmark.

1. Use one8-H100 development run. First demonstrate that the implementation can overfit a fixed two-bit constraint system, then increase to four and eight bits. Keep the public-condition/target boundary explicit. A fixed or aligned coordinate serializer can distinguish binding/position difficulty from the parity operation itself. These are changed task distributions and must be named separately from the original held-out-structure pilot.
2. Compare the original sampled-target loss to a controlled Rao–Blackwellized oracle-target loss. The exact synthetic oracle supplies conditional probabilities0,1/2,1 at each masked position. This is privileged synthetic supervision; it is a diagnostic control, not a generally available natural-language training method.
3. Use development data to choose learning rate, update budget and any curriculum. Log deterministic-bit proper loss and fair-bit calibration, not only thresholded accuracy. Before locking a fresh test, aim for an informative schedule-specific KL bound; for example, exact information-set endpoint KL<=0.1nats on development certifies valid mass>=exp(-0.1)=90.48% on those same audited conditions. That development certificate does not transfer automatically to unseen conditions.
4. Freeze the successful recipe and evaluate three independent training seeds on a fresh declared test panel. Retain all outcomes. Match actual network calls for information-set versus random-halves policies, and keep exact-oracle and uniform-predictor controls.
5. Only after this controlled mechanism is measurable should longer contexts or ordinary-language tasks be added. A learned dependency selector, realistic representation bottleneck or substantive new tradeoff would still be needed for a stronger research contribution. Exact rank-based scheduling alone is an oracle construction.

## Why the oracle-target control is mathematically appropriate

Let Z=(A,C,X_t,t) denote the public condition and corrupted canvas. For the existing loss, write g(Y,Z;theta) for its sampled-target gradient. Replace each masked one-hot label with p_i=P(Y_i|Z), giving the soft-target cross entropy with unchanged8/(t*N) weighting and unchanged empty-mask law. Since the model's Jacobian is fixed conditional on Z,

    g_RB(Z;theta) = E[g(Y,Z;theta) | Z].

The expected objective gradient is unchanged, while the law of total covariance gives

    Cov(g) - Cov(g_RB) = E[Cov(g|Z)] >= 0

in positive-semidefinite order. This is the classical Rao–Blackwell variance reduction argument, not a new theorem or a guarantee of faster nonconvex optimization. It removes avoidable label noise from oracle-fair coordinates and helps isolate whether the network can learn the deterministic conditionals. It must not be described as a learned posterior when the exact symbolic oracle supplied the training targets.

The original pilot used hard sampled targets throughout; no oracle-target updates are included in its three checkpoints. This document describes follow-up development, not additional executed experiments.
