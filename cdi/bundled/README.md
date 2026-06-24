# Bundled Dependencies

This directory contains small project-local dependencies vendored into CDI for
release reproducibility.

- `leanbase/`: local training/config/checkpoint helper library used by CDI.
- `diffusion/`: diffusion-process utilities needed by `cdi`.

Do not add large experiment packages here unless they are required by the
public CBED/CDI reproduction path.

Each bundled component carries its own license file in its package directory.
