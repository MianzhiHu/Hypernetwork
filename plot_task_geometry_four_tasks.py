from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Descriptive task geometry, not a fitted clustering or significance analysis.
# Pairwise first CCA uses a different alignment for each task pair.
# Treat the resulting distances as descriptive dissimilarities, not shared axes.
selected_config = 'layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_8_reg_0.0'
source = Path('./Results/HyperNN_Grid_Search_Evaluation/Cross_Task_Convergence/cca_by_task_pair.csv')
output = Path('./Figures/Task_Geometry_Four_Tasks') / selected_config / 'CCA'
names = ['cct', 'dd', 'motor', 'stopsignal']
labels = ['CCT', 'DD', 'Motor', 'Stop Signal']
data = pd.read_csv(source)
data = data.loc[data.config_id.eq(selected_config)].copy()
assert len(data) == 6 and not data.duplicated(['task_a', 'task_b']).any()
similarity = np.eye(4)
for row in data.itertuples():
    i, j = names.index(row.task_a), names.index(row.task_b)
    similarity[i, j] = similarity[j, i] = row.first_cca
assert np.isfinite(similarity).all() and (similarity >= -1e-8).all() and (similarity <= 1 + 1e-8).all()
# Classical MDS on dissimilarity sqrt(2*(1-first CCA r)).
# Check negative eigenvalues: pairwise CCA need not yield Euclidean distances.
center = np.eye(4) - np.ones((4, 4)) / 4
gram = center @ similarity @ center
values, vectors = np.linalg.eigh(gram)
order = np.argsort(values)[::-1]
values, vectors = values[order], vectors[:, order]
negative_inertia = -values[values < -1e-10].sum()
coordinates = vectors[:, :2] * np.sqrt(np.maximum(values[:2], 0))
retained = values[:2].sum() / values[values > 1e-10].sum()
fractions = values[:2] / values[values > 1e-10].sum()
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'task': names, 'axis_1': coordinates[:, 0], 'axis_2': coordinates[:, 1]}).to_csv(output / 'task_coordinates.csv', index=False)
pd.DataFrame(similarity, index=names, columns=names).to_csv(output / 'task_cca.csv')
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial', 'DejaVu Sans'],
                     'font.size':10, 'svg.fonttype':'none', 'pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(7.2, 5.8), layout='constrained')
ax.scatter(coordinates[:, 0], coordinates[:, 1], s=75, color='#39739D', edgecolor='white', linewidth=0.8, zorder=3)
for label, (x, y) in zip(labels, coordinates):
    offset = (7, 7)
    ax.annotate(label, (x, y), xytext=offset, textcoords='offset points', fontsize=10)
ax.axhline(0, color='0.85', linewidth=0.7, zorder=0)
ax.axvline(0, color='0.85', linewidth=0.7, zorder=0)
ax.set_aspect('equal', adjustable='box')
ax.margins(.25)
ax.spines[['top','right']].set_visible(False)
ax.set(xlabel=f'MDS axis 1 ({fractions[0]:.1%})', ylabel=f'MDS axis 2 ({fractions[1]:.1%})',
       title=f'Task similarity: four-task HyperNN analysis\nFirst-component CCA; 2D retains {retained:.1%} of positive inertia')
fig.savefig(output / 'task_map.png', dpi=300)
fig.savefig(output / 'task_map.svg')
fig.savefig(output / 'task_map.pdf')
plt.close(fig)
(output / 'README.txt').write_text(f"Source: {source}\nConfiguration: {selected_config}\nFour tasks; all 6 pairs retained; 92 participants. Within-task GCCA consensus across ten trial rotations.\nClassical MDS on distance sqrt(2*(1-first CCA r)); two dimensions retain {retained:.6f} of positive inertia. Negative eigenvalue mass: {negative_inertia:.6f}.\nAxes have no assigned psychological meaning; no cluster assignments, significance claims, or uncertainty intervals.\nProjection can distort distances: consult task_cca.csv.\nPNG is a screen preview; PDF/SVG are vector exports.\n")
print('Retained:', retained, 'Negative inertia:', negative_inertia)
print(pd.DataFrame(coordinates, index=labels).round(3).to_string())
print(output / 'task_map.png')

print(data[['task_a', 'task_b', 'first_cca']].sort_values('first_cca', ascending=False).to_string(index=False))
