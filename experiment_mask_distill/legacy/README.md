Legacy experimental logic is disabled by default.

This includes:

- old SourceGate MLP training
- TextFreeMV supervision
- open-reliability gate targets
- GT upper-bound gate targets
- dual-branch diagnostic probe
- projected-sem probe logging

The current repository still keeps some of this logic in
`experiment_mask_distill/trainer_mask_distill.py` behind disabled config flags,
so historical ablations remain reproducible.

Default final-method configs should not enable these paths.
