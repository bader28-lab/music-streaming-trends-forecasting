"""
Assemble le support final : insere la bibliotheque Plotly et les figures JSON dans le gabarit.
Le fichier produit est autonome (fonctionne sans connexion, hors police d'ecriture).

Execution : python build_figures.py puis python assembler_support.py
"""
import json
import os
from pathlib import Path

import plotly

gabarit = Path('support_template.html').read_text()
plotly_js = Path(os.path.dirname(plotly.__file__), 'package_data', 'plotly.min.js').read_text()
figures = {f.stem: json.loads(f.read_text()) for f in sorted(Path('figures').glob('*.json'))}

page = (gabarit.replace('/*PLOTLY_JS*/', plotly_js)
               .replace('/*FIGURES_JSON*/', json.dumps(figures)))

sortie = Path('support_soutenance.html')
sortie.write_text(page)
print(f"{sortie} : {sortie.stat().st_size // 1024} Ko, {len(figures)} figures")
