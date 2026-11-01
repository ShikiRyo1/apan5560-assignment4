# Executed verification

November 1, 2026. The final 10-epoch EBM and 30-epoch diffusion checkpoints
were built into a new Linux CPU Docker container. No host directories were
mounted; the container loaded all four included course checkpoints.

- 6 core pytest tests passed: architecture/gradients, numerical time encoding,
  sampler input updates without weight-gradient mutation, energy descent on a
  controlled quadratic, noising identity, and Questions 6–9 autograd behavior.
- 26 independent mathematical/implementation checks passed, including DDIM
  oracle cases and U-Net downsampling dimensions.
- 240 independent HTTP/container checks passed against the final source build:
  old endpoints; both new model metadata endpoints; actual PNG generation and
  RGB 32×32 decoding; grid pixel agreement; fixed and returned random seeds;
  count/step boundary cases; malformed requests; checkpoint SHA-256 agreement;
  zero host mounts and zero container restarts.
- Both models completed every planned epoch using all 50,000 training images.
  `TRAINING_REPORT.md` records post-training diagnostics and visible limitations.
- All four pages of the answers PDF were rendered and visually reviewed;
  all nine questions and their subparts are covered.

The submission archive receives an additional fresh-extraction Docker build
and HTTP check recorded in the local delivery audit. These are local checks,
not a claim that the instructor has run the application or awarded a grade.

The training was GPU-based, but the deployed container uses the locked CPU
runtime. Same-environment seeded generation was verified; bit-identical GPU,
CPU, operating-system or version changes are not guaranteed.
