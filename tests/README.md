How to run tests

1) Activate the conda env for this project:

```
conda activate ephys-gpt
```

2) Run the suite:

```
pytest -q
```

Notes
- Tests use tiny configs and synthetic tensors to keep runtime low.
- Some models require CPU-only by default; no GPU is assumed.
- The dataset tests use temporary directories and write small .npy files.


