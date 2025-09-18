# Force Field Heatmap + Vectors

This figure shows the magnitude of the pedestrian social-force field as a heatmap with a quiver overlay for direction.

## How to Generate

```
uv run python results/figures/fig_force_field.py \
  --png docs/img/fig-force-field.png \
  --pdf docs/figures/fig-force-field.pdf
```

- PNG is suitable for docs/web; PDF is LaTeX-ready (tight bbox, pdf.fonttype=42).
- Headless-safe (Agg), reproducible outputs.

## Notes
- Uses `FastPysfWrapper` around the PySocialForce simulator to sample forces on a grid.
- Subsamples quiver to keep the vector field readable.
