# Force Field Heatmap + Vectors

This figure shows the magnitude of the pedestrian social-force field as a heatmap with a quiver overlay for direction.

## How to Generate

```
uv run python -m robot_sf.benchmark.figures.force_field \
  --png docs/img/fig-force-field.png \
  --pdf docs/figures/fig-force-field.pdf
```

- PNG is suitable for docs/web; PDF is LaTeX-ready (tight bbox, pdf.fonttype=42).
- Headless-safe (Agg), reproducible outputs.

## Notes
- Uses `FastPysfWrapper` around the PySocialForce simulator to sample forces on a grid.
- Subsamples quiver to keep the vector field readable.

## Include in LaTeX

Copy-paste one of these into your paper. Prefer the vector PDF for print quality:

```latex
% In your preamble:
% \usepackage{graphicx}

% Single-column width
\begin{figure}[t]
  \centering
  \includegraphics[width=0.48\textwidth]{docs/figures/fig-force-field.pdf}
  \caption{Social-force field magnitude (heatmap) with direction (quiver).}
  \label{fig:force-field}
\end{figure}
```

Two-column layout (spanning):

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.9\textwidth]{docs/figures/fig-force-field.pdf}
  \caption{Social-force field visualization.}
  \label{fig:force-field-wide}
\end{figure*}
```

Notes
- Exports use LaTeX-friendly Matplotlib settings (tight bbox, `pdf.fonttype=42`).
- You can set a graphics path in your LaTeX preamble: `\graphicspath{{docs/figures/}}`.
