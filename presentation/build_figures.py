"""
Construit les graphiques Plotly du support de presentation P9 et les exporte en JSON.

Les calculs reprennent la logique des deux notebooks livrables (memes nettoyages,
memes parametres, meme graine aleatoire pour Prophet). Les chiffres cles sont
affiches en fin d'execution pour etre compares aux sorties des notebooks.

Execution : python build_figures.py   (depuis le dossier P9/presentation)
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

logging.getLogger('cmdstanpy').disabled = True

DATA = Path('../data')
SORTIE = Path('figures')
SORTIE.mkdir(exist_ok=True)

# Charte du support (style "Studio")
SOMBRE, ACCENT, GRIS, AMBRE = '#0d1321', '#3a6ff7', '#8b93a7', '#f2a541'
POLICE = dict(family='Manrope, sans-serif', size=17, color=SOMBRE)


def mise_en_forme(fig, **kwargs):
    """Applique la charte commune a toutes les figures."""
    fig.update_layout(font=POLICE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10, r=20, t=10, b=10), showlegend=False,
                      hoverlabel=dict(font_family='Manrope'), separators=', ', **kwargs)
    fig.update_xaxes(gridcolor='#e3e6ec', zerolinecolor='#c9ced8', linecolor='#c9ced8')
    fig.update_yaxes(gridcolor='#e3e6ec', zerolinecolor='#c9ced8', linecolor='#c9ced8')
    return fig


def exporter(fig, nom):
    (SORTIE / f'{nom}.json').write_text(fig.to_json())


chiffres = {}

# ======================================================================
# MISSION 1 : catalogue
# ======================================================================
df_genre = pd.read_csv(DATA / 'spotify-tracks-dataset-detailed.csv').dropna().drop_duplicates()
df_tracks = df_genre.drop_duplicates(subset='track_id').copy()
df_tracks['duration_sec'] = df_tracks['duration_ms'] / 1000
chiffres['lignes df_genre / df_tracks'] = (len(df_genre), len(df_tracks))

# Distribution de la popularite (df_genre, base principale comme le corrige)
repartition = df_genre['popularity'].value_counts().sort_index()
chiffres['popularite nulle (%) / mediane / moyenne'] = (round((df_genre['popularity'] == 0).mean() * 100, 1),
                                                       df_genre['popularity'].median(), round(df_genre['popularity'].mean(), 1))
fig = go.Figure(go.Bar(x=repartition.index, y=repartition.values,
                       marker_color=[AMBRE if v == 0 else ACCENT for v in repartition.index],
                       hovertemplate='popularite %{x} : %{y} lignes<extra></extra>'))
mise_en_forme(fig, xaxis=dict(title='popularité (0 à 100)'), yaxis=dict(title='nombre de lignes'), bargap=0.1)
exporter(fig, 'm1_distribution_popularite')

# Classements d'artistes (df_tracks, featurings eclates, seuil de 8 morceaux)
artistes = df_tracks.assign(artists=df_tracks['artists'].str.split(';')).explode('artists')
artistes['artists'] = artistes['artists'].str.strip()
stats = artistes.groupby('artists')['popularity'].agg(['mean', 'count'])
top_presence = stats.sort_values('count', ascending=False).head(15)
top_pop = stats[stats['count'] >= 8].sort_values('mean', ascending=False).head(15)
chiffres['artistes communs classements 1 et 2'] = len(set(top_presence.index) & set(top_pop.index))
chiffres['n1 presence / n1 popularite'] = (top_presence.index[0], int(top_presence['count'].iloc[0]),
                                            top_pop.index[0], round(top_pop['mean'].iloc[0], 1))

# Meme classement calcule sur df_genre : un morceau range dans plusieurs genres y compte plusieurs fois
from plotly.subplots import make_subplots
artistes_g = df_genre.assign(artists=df_genre['artists'].str.split(';')).explode('artists')
artistes_g['artists'] = artistes_g['artists'].str.strip()
stats_g = artistes_g.groupby('artists')['popularity'].agg(['mean', 'count'])
top_pop_g = stats_g[stats_g['count'] >= 8].sort_values('mean', ascending=False).head(15)
top_pop_g['reels'] = stats['count'].reindex(top_pop_g.index)
chiffres['df_genre : sous le seuil en reel / communs'] = (int((top_pop_g['reels'] < 8).sum()),
                                                         len(set(top_pop_g.index) & set(top_pop.index)))

t10 = top_pop.head(10).iloc[::-1]
g10 = top_pop_g.head(10).iloc[::-1]
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.28,
                    subplot_titles=('Compté une fois', 'Compté une fois par genre'))
fig.add_trace(go.Bar(x=t10['mean'], y=t10.index, orientation='h', marker_color=ACCENT,
                     texttemplate='%{x:.1f}', textposition='outside', customdata=t10['count'],
                     hovertemplate='%{y} : %{x:.1f} (%{customdata} morceaux)<extra></extra>'), row=1, col=1)
fig.add_trace(go.Bar(x=g10['mean'], y=g10.index, orientation='h',
                     marker_color=[AMBRE if r < 8 else ACCENT for r in g10['reels']],
                     texttemplate='%{x:.1f}', textposition='outside',
                     customdata=g10[['count', 'reels']].values,
                     hovertemplate='%{y} : %{x:.1f}<br>%{customdata[0]} lignes comptées, %{customdata[1]} morceaux réels<extra></extra>'),
              row=1, col=2)
mise_en_forme(fig)
fig.update_xaxes(range=[60, 90], title_text='popularité moyenne')
fig.update_layout(margin=dict(l=10, r=20, t=40, b=10))
fig.update_annotations(font=dict(family='Manrope', size=17, color=SOMBRE))
exporter(fig, 'm1_artistes_populaires')

# Trois variables audio liees : energy contre loudness, couleur = acousticness
from scipy.stats import spearmanr, kruskal, chi2_contingency
chiffres['Spearman energy/loudness'] = round(spearmanr(df_genre['energy'], df_genre['loudness'])[0], 3)
chiffres['Spearman acousticness/energy'] = round(spearmanr(df_genre['acousticness'], df_genre['energy'])[0], 3)
echantillon = df_genre.sample(3000, random_state=42)
fig = go.Figure(go.Scatter(x=echantillon['energy'], y=echantillon['loudness'], mode='markers',
                           marker=dict(size=5, opacity=0.55, color=echantillon['acousticness'],
                                       colorscale=[[0, ACCENT], [1, AMBRE]], showscale=True,
                                       colorbar=dict(title='acousticness', thickness=12, len=0.8)),
                           hovertemplate='energy %{x:.2f}<br>loudness %{y:.1f} dB<extra></extra>'))
mise_en_forme(fig, xaxis=dict(title='energy'), yaxis=dict(title='loudness (dB)'))
exporter(fig, 'm1_audio_lie')

# Annexe : matrice de correlation (df_genre, memes 11 variables que le notebook)
corr_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
             'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_sec', 'popularity']
corr_matrix = df_genre.assign(duration_sec=df_genre['duration_ms'] / 1000)[corr_cols].corr()
fig = go.Figure(go.Heatmap(z=corr_matrix.values, x=corr_cols, y=corr_cols, zmin=-1, zmax=1,
                           colorscale='RdBu_r', texttemplate='%{z:.2f}', textfont=dict(size=11),
                           colorbar=dict(thickness=10, len=0.9),
                           hovertemplate='%{y} / %{x} : %{z:.2f}<extra></extra>'))
mise_en_forme(fig, xaxis=dict(tickangle=-45, tickfont=dict(size=13), showgrid=False),
              yaxis=dict(autorange='reversed', tickfont=dict(size=13), showgrid=False))
exporter(fig, 'annexe_matrice_correlation')

# Annexe : radar des trois genres compares (choix de Bader : k-pop, sleep, drum-and-bass)
genres_compares = ['k-pop', 'sleep', 'drum-and-bass']
radar_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
profil = df_genre[df_genre['track_genre'].isin(genres_compares)].groupby('track_genre')[radar_cols].mean().reindex(genres_compares)
fig = go.Figure()
for genre, couleur in zip(genres_compares, [ACCENT, GRIS, AMBRE]):
    fig.add_trace(go.Scatterpolar(r=list(profil.loc[genre]) + [profil.loc[genre].iloc[0]],
                                  theta=radar_cols + [radar_cols[0]], name=genre, fill='toself', opacity=0.5,
                                  line=dict(color=couleur, width=2),
                                  hovertemplate=genre + ' · %{theta} : %{r:.2f}<extra></extra>'))
mise_en_forme(fig, legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
              polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(range=[0, 1], tickfont=dict(size=12)),
                         angularaxis=dict(tickfont=dict(size=13))))
fig.update_layout(showlegend=True, margin=dict(l=70, r=70, t=30, b=30))
exporter(fig, 'annexe_radar_genres')

# Popularite par genre : les 8 genres les plus et les 8 moins populaires (mediane)
groupes = [g['popularity'].values for _, g in df_genre.groupby('track_genre')]
H, p = kruskal(*groupes)
k, n = len(groupes), len(df_genre)
chiffres['Kruskal H / epsilon2'] = (round(H), round((H - k + 1) / (n - k), 3))
medianes = df_genre.groupby('track_genre')['popularity'].median().sort_values()
selection = list(medianes.index[:8]) + list(medianes.index[-8:])
fig = go.Figure()
for genre in selection:
    valeurs = df_genre.loc[df_genre['track_genre'] == genre, 'popularity']
    q1, med, q3 = valeurs.quantile([.25, .5, .75])
    fig.add_trace(go.Box(x=[genre], name=genre, q1=[q1], median=[med], q3=[q3],
                         lowerfence=[max(valeurs.min(), q1 - 1.5 * (q3 - q1))],
                         upperfence=[min(valeurs.max(), q3 + 1.5 * (q3 - q1))],
                         marker_color=ACCENT if genre in medianes.index[-8:] else GRIS, line_width=1.5))
mise_en_forme(fig, yaxis=dict(title='popularité'), xaxis=dict(tickangle=-40))
exporter(fig, 'm1_popularite_genres')

# Contenu explicite par genre : les 8 plus et les 8 moins explicites
contingence = pd.crosstab(df_genre['track_genre'], df_genre['explicit'])
chi2, p_chi2, ddl, _ = chi2_contingency(contingence)
v_cramer = np.sqrt(chi2 / (n * (min(contingence.shape) - 1)))
chiffres['khi2 / ddl / V de Cramer'] = (round(chi2), ddl, round(v_cramer, 3))
taux = (df_genre.groupby('track_genre')['explicit'].mean() * 100).sort_values()
extremes = pd.concat([taux.head(8), taux.tail(8)])
chiffres['taux explicite min / max'] = ((taux.index[0], round(taux.iloc[0], 1)), (taux.index[-1], round(taux.iloc[-1], 1)))
fig = go.Figure(go.Bar(x=extremes.index, y=extremes.values, texttemplate='%{y:.1f} %', textposition='outside',
                       marker_color=[GRIS] * 8 + [ACCENT] * 8,
                       hovertemplate='%{x} : %{y:.1f} %<extra></extra>'))
mise_en_forme(fig, yaxis=dict(title='% de morceaux explicites', range=[0, extremes.max() * 1.15]), xaxis=dict(tickangle=-40))
exporter(fig, 'm1_explicite_genres')

# Part de la popularite expliquee : genre (epsilon2) contre variables audio (r2)
variables = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
             'liveness', 'valence', 'tempo', 'duration_sec', 'time_signature']
r = df_tracks[variables + ['popularity']].corr()['popularity'].drop('popularity')
part = (r ** 2 * 100).sort_values(ascending=False).head(5)
part = pd.concat([pd.Series({'genre': (H - k + 1) / (n - k) * 100}), part]).iloc[::-1]
chiffres['r instrumentalness'] = round(r['instrumentalness'], 3)
fig = go.Figure(go.Bar(x=part.values, y=part.index, orientation='h',
                       marker_color=[ACCENT if i == 'genre' else GRIS for i in part.index],
                       texttemplate='%{x:.1f} %', textposition='outside',
                       hovertemplate='%{y} : %{x:.1f} %<extra></extra>'))
mise_en_forme(fig, xaxis=dict(title='part de la popularité expliquée (%)', range=[0, 32]))
exporter(fig, 'm1_genre_contre_son')

# ======================================================================
# MISSION 2 : series temporelles (pop, classement mondial)
# ======================================================================
GENRE, MARCHE, JOURS_TEST, HORIZON_JOURS = 'pop', 'Global', 90, 182
np.random.seed(42)

classements = pd.read_csv(DATA / 'chart_tracks_genre_2017_2020.csv', usecols=['date', 'country', 'genre'])
classements['date'] = pd.to_datetime(classements['date'], format='%d/%m/%Y')
marche = classements[classements['country'] == MARCHE]

serie = marche[marche['genre'] == GENRE].groupby('date').size()
totale = marche.groupby('date').size()
renseignee = marche.dropna(subset=['genre']).groupby('date').size()
grille = pd.date_range(totale.index.min(), totale.index.max(), freq='D')
serie, totale, renseignee = (s.reindex(grille, fill_value=0) for s in (serie, totale, renseignee))
serie = serie.astype(float)
serie[(totale == 0) | ((totale > 0) & (renseignee == 0))] = np.nan   # trous de collecte

moyenne_mobile = serie.rolling(30, min_periods=1).mean()
chiffres['moyenne annuelle 2017 / 2020'] = tuple(serie.groupby(serie.index.year).mean().round(0).loc[[2017, 2020]])

fig = go.Figure()
fig.add_trace(go.Scatter(x=serie.index, y=serie, line=dict(color=GRIS, width=1), opacity=0.5,
                         hovertemplate='%{x|%d/%m/%Y} : %{y} morceaux<extra></extra>'))
fig.add_trace(go.Scatter(x=moyenne_mobile.index, y=moyenne_mobile, line=dict(color=ACCENT, width=3),
                         hovertemplate='moyenne 30 jours : %{y:.0f}<extra></extra>'))
mise_en_forme(fig, yaxis=dict(title='morceaux pop dans le Top 200'), xaxis=dict(tickformat='%m/%Y'))
exporter(fig, 'm2_serie_pop')

# Prophet, meme sequence que le notebook (graine, modele de test, essais, modele final)
serie_prophet = pd.DataFrame({'ds': serie.index, 'y': serie.values})
entrainement, test = serie_prophet.iloc[:-JOURS_TEST], serie_prophet.iloc[-JOURS_TEST:]
modele = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(entrainement)
prevision = modele.predict(modele.make_future_dataframe(periods=JOURS_TEST))
prevu = prevision[prevision['ds'].isin(test['ds'])]['yhat'].values
erreur = test['y'].values - prevu
chiffres['MAE / RMSE / MAPE'] = (round(np.mean(np.abs(erreur)), 1), round(np.sqrt(np.mean(erreur ** 2)), 1),
                                 round(np.mean(np.abs(erreur) / test['y'].values) * 100, 1))
chiffres['jours sous-estimes'] = int((erreur > 0).sum())

from plotly.subplots import make_subplots
noms_jours = ['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim']
effet_jour = prevision.groupby(prevision['ds'].dt.dayofweek)['weekly'].mean()
annuelle = prevision.set_index('ds')['yearly']['2019']
chiffres['effet vendredi / samedi'] = (round(effet_jour[4], 1), round(effet_jour[5], 1))
chiffres['effet annuel max / min'] = ((round(annuelle.max(), 1), annuelle.idxmax().strftime('%d/%m')),
                                      (round(annuelle.min(), 1), annuelle.idxmin().strftime('%d/%m')))
fig = make_subplots(rows=1, cols=2, column_widths=[0.35, 0.65], horizontal_spacing=0.1,
                    subplot_titles=('Effet du jour de la semaine', "Effet de la période de l'année"))
fig.add_trace(go.Bar(x=noms_jours, y=effet_jour.values, marker_color=[ACCENT] * 4 + [AMBRE] * 2 + [ACCENT],
                     hovertemplate='%{x} : %{y:+.1f} morceaux<extra></extra>'), row=1, col=1)
fig.add_trace(go.Scatter(x=annuelle.index, y=annuelle.values, line=dict(color=ACCENT, width=3),
                         hovertemplate='%{x|%d/%m} : %{y:+.1f} morceaux<extra></extra>'), row=1, col=2)
fig.add_hline(y=0, line=dict(color=GRIS, width=1))
mise_en_forme(fig, yaxis=dict(title='morceaux pop en plus ou en moins'), xaxis2=dict(tickformat='%d/%m'))
fig.update_annotations(font=dict(family='Manrope', size=17, color=SOMBRE))
fig.update_layout(margin=dict(l=10, r=20, t=40, b=10))
fig.update_yaxes(range=[effet_jour.min() * 1.6, effet_jour.max() * 2], row=1, col=1)
exporter(fig, 'm2_composantes')

fig = go.Figure()
recent = entrainement[entrainement['ds'] >= '2020-03-01']
fig.add_trace(go.Scatter(x=recent['ds'], y=recent['y'], line=dict(color=GRIS, width=1.5),
                         hovertemplate='%{x|%d/%m} : %{y}<extra>entrainement</extra>'))
fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], line=dict(color=SOMBRE, width=2.5),
                         hovertemplate='%{x|%d/%m} : %{y}<extra>reel</extra>'))
fig.add_trace(go.Scatter(x=test['ds'], y=prevu, line=dict(color=ACCENT, width=2.5, dash='dash'),
                         hovertemplate='%{x|%d/%m} : %{y:.0f}<extra>prevu</extra>'))
fig.add_vline(x=test['ds'].min(), line=dict(color=GRIS, dash='dot'))
mise_en_forme(fig, yaxis=dict(title='morceaux pop'), xaxis=dict(tickformat='%d/%m'))
exporter(fig, 'm2_test')

for plage in [0.8, 0.9, 0.95]:   # memes essais que le notebook, pour conserver l'etat aleatoire
    np.random.seed(42)
    Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
            changepoint_range=plage).fit(entrainement).predict(test[['ds']])

modele_final = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(serie_prophet)
finale = modele_final.predict(modele_final.make_future_dataframe(periods=HORIZON_JOURS))
horizon = finale[finale['ds'] > serie_prophet['ds'].max()]
largeur = horizon['yhat_upper'] - horizon['yhat_lower']
chiffres['prevision moyenne / Noel / largeur 1er et 6e mois'] = (
    round(horizon['yhat'].mean()), round(horizon.set_index('ds').loc['2020-12-25', 'yhat']),
    round(largeur.head(30).mean(), 1), round(largeur.tail(30).mean(), 1))

historique = serie_prophet[serie_prophet['ds'] >= '2019-11-01']
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.concat([horizon['ds'], horizon['ds'][::-1]]),
                         y=pd.concat([horizon['yhat_upper'], horizon['yhat_lower'][::-1]]),
                         fill='toself', fillcolor='rgba(58,111,247,0.15)', line=dict(width=0), hoverinfo='skip'))
fig.add_trace(go.Scatter(x=historique['ds'], y=historique['y'], line=dict(color=SOMBRE, width=1.5),
                         hovertemplate='%{x|%d/%m/%Y} : %{y}<extra>reel</extra>'))
fig.add_trace(go.Scatter(x=horizon['ds'], y=horizon['yhat'], line=dict(color=ACCENT, width=3),
                         hovertemplate='%{x|%d/%m/%Y} : %{y:.0f}<extra>prevu</extra>'))
mise_en_forme(fig, yaxis=dict(title='morceaux pop'), xaxis=dict(tickformat='%m/%Y'))
exporter(fig, 'm2_prevision')

# Cadence des sorties : moyenne par mois calendaire
sorties = pd.read_csv(DATA / 'release_dates_genre_2017_2020.csv', usecols=['Release_date', 'Genre_new'])
sorties['Release_date'] = pd.to_datetime(sorties['Release_date'], format='%Y-%m-%d')
serie_sorties = sorties[sorties['Genre_new'] == GENRE].set_index('Release_date').resample('MS').size()
par_mois = serie_sorties.groupby(serie_sorties.index.month).mean()
noms_mois = ['janv.', 'févr.', 'mars', 'avr.', 'mai', 'juin', 'juil.', 'août', 'sept.', 'oct.', 'nov.', 'déc.']
chiffres['sorties moyennes nov / dec'] = (round(par_mois[11]), round(par_mois[12]))
fig = go.Figure(go.Bar(x=noms_mois, y=par_mois.values, texttemplate='%{y:.0f}', textposition='outside',
                       marker_color=[GRIS] * 10 + [ACCENT, AMBRE],
                       hovertemplate='%{x} : %{y:.0f} sorties en moyenne<extra></extra>'))
mise_en_forme(fig, yaxis=dict(title='sorties pop par mois (moyenne)', range=[0, par_mois.max() * 1.15]))
exporter(fig, 'm2_sorties_mois')

# Comparaison entre marches : evolution de la part de la pop, janvier a octobre 2017 contre 2020
comparables = classements.dropna(subset=['genre'])
comparables = comparables[(comparables['date'].dt.month <= 10) & (comparables['date'].dt.year.isin([2017, 2020]))]
part_genre = (comparables.assign(est=comparables['genre'] == GENRE, annee=comparables['date'].dt.year)
              .groupby(['country', 'annee'])['est'].mean().unstack() * 100)
NOMS_FR = {'Argentina': 'Argentine', 'Australia': 'Australie', 'Austria': 'Autriche', 'Belgium': 'Belgique',
            'Brazil': 'Brésil', 'Canada': 'Canada', 'Chile': 'Chili', 'Colombia': 'Colombie', 'Costa Rica': 'Costa Rica',
            'Denmark': 'Danemark', 'Ecuador': 'Équateur', 'Finland': 'Finlande', 'France': 'France', 'Germany': 'Allemagne',
            'Global': 'MONDIAL', 'Indonesia': 'Indonésie', 'Ireland': 'Irlande', 'Italy': 'Italie', 'Malaysia': 'Malaisie',
            'Mexico': 'Mexique', 'Netherlands': 'Pays-Bas', 'New Zealand': 'Nouvelle-Zélande', 'Norway': 'Norvège',
            'Peru': 'Pérou', 'Philippines': 'Philippines', 'Poland': 'Pologne', 'Portugal': 'Portugal', 'Singapore': 'Singapour',
            'Spain': 'Espagne', 'Sweden': 'Suède', 'Switzerland': 'Suisse', 'Taiwan': 'Taiwan', 'Turkey': 'Turquie',
            'UK': 'Royaume-Uni', 'USA': 'États-Unis'}
evolution = (part_genre[2020] - part_genre[2017]).sort_values(ascending=False)
chiffres['marches en baisse'] = f"{(evolution < 0).sum()} / {len(evolution)}"
fig = go.Figure(go.Bar(x=evolution.values, y=[NOMS_FR.get(m, m) for m in evolution.index], orientation='h',
                       marker_color=[AMBRE if m == MARCHE else (ACCENT if v < 0 else GRIS) for m, v in evolution.items()],
                       hovertemplate='%{y} : %{x:+.1f} points<extra></extra>'))
mise_en_forme(fig, xaxis=dict(title='évolution de la part de la pop (points)'), yaxis=dict(tickfont=dict(size=13), dtick=1), bargap=0.25)
exporter(fig, 'm2_marches')

print(json.dumps(chiffres, indent=1, ensure_ascii=False, default=str))
