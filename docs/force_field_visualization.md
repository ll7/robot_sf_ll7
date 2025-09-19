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
- PNGs: `docs/img/force_field_example.png`, `docs/img/force_field_example_norm.png`
- PDFs: `docs/figures/force_field_example.pdf`, `docs/figures/force_field_example_norm.pdf`

## Tips
- For LaTeX inclusion, prefer vector PDFs. The scripts export `.pdf` next to the PNGs with LaTeX-friendly rcParams (bbox tight, pdf.fonttype=42, font sizes 9/8).
- Subsample quiver arrows for clarity (see `step` parameter in the normalized script).
- Use consistent colormaps and labels. Include units in axis labels.

### Include in LaTeX

```latex
% Preamble
% \usepackage{graphicx}
% Optional: \graphicspath{{docs/figures/}}

\begin{figure}[t]
	\centering
	\includegraphics[width=0.48\textwidth]{docs/figures/force_field_example_norm.pdf}
	\caption{Normalized social-force vector field with magnitude colormap.}
	\label{fig:ff-example}
\end{figure}
```
