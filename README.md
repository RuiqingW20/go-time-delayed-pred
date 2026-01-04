We adopt a time-based, protein-disjoint data split to reflect the evolving nature of GO annotations.
Since GO terms are continuously added as new experimental evidence emerges, random splits may introduce temporal leakage by allowing models to access information that would not be available at prediction time.
Our evaluation simulates a realistic deployment scenario, where models trained on historical annotations at time t0 are tasked with predicting GO terms newly assigned in a future time window (t0, t0+Δt].
