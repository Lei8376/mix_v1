# Cursor Project Rules

## General
- Language: Python
- Python version >= 3.8
- Use PyTorch as the deep learning framework
- Avoid unnecessary abstractions

## Code Style
- Follow PEP8
- Use explicit variable names
- Prefer dataclasses for configs
- Avoid global state

## ML / Research
- Keep model, dataset, and trainer separated
- Log training metrics explicitly
- Do not generate fake experimental results
- Prefer reproducibility over clever tricks

## Comments
- Explain WHY, not WHAT
- Add math explanation when implementing loss functions