# Force Field Visualization

This guide shows how to generate reproducible force field figures for documentation and papers. Figures are created from code and saved under `docs/img/` as recommended in the dev guide.

## Examples

Two example scripts are available:
- `examples/plot_force_field.py` — interactive demo using quiver (opens a window)
- `examples/plot_force_field_save.py` — non-interactive, saves `docs/img/force_field_example.png`
- `examples/plot_force_field_normalized.py` — normalized arrows with magnitude colormap; saves `docs/img/force_field_example_norm.png`

The examples construct a small `pysocialforce.Simulator`, wrap it via `FastPysfWrapper`, sample a grid, and plot the vector field.

## Reproducible figure generation

These scripts run headless by default when used non-interactively. To regenerate the figures:

```
# Save standard quiver PNG
uv run python examples/plot_force_field_save.py

# Save normalized quiver with magnitude colormap
uv run python examples/plot_force_field_normalized.py
```

Outputs:
- `docs/img/force_field_example.png`
- `docs/img/force_field_example_norm.png`

## Tips
- For LaTeX inclusion, prefer vector PDFs when possible. You can adapt the scripts to also save `.pdf` by calling `fig.savefig(".../file.pdf")` and following the rcParams guidance in the dev guide (bbox tight, pdf.fonttype=42, font sizes 9/8).
- Subsample quiver arrows for clarity (see `step` parameter in the normalized script).
- Use consistent colormaps and labels. Include units in axis labels.
