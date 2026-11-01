# Presentation

Slides are in French.

- `support_soutenance.html`: standalone presentation (Plotly embedded, works offline except for the font). Navigation: arrow keys, space, mouse wheel or the dots on the right. 21 slides, then 7 appendices for the Q&A.
- `support_soutenance.pdf`: PDF version.
- `support_template.html`: slide template (text and layout).
- `build_figures.py`: recomputes the 14 charts from `../data/` with the same logic as the notebooks, and prints the key figures for checking.
- `figures/`: charts exported as JSON.
- `assembler_support.py`: inserts Plotly and the charts into the template to produce the presentation.

To rebuild after a change:

```bash
python build_figures.py && python assembler_support.py
```
