from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('Results/HyperNN_Grid_Search_Evaluation')
out=Path('Figures/Uniform_Prediction'); out.mkdir(parents=True,exist_ok=True)
x=pd.read_csv(root/'test_results_all.csv')
tasks=['ant','cct','dd','motor','stroop','dpx','stopsignal','twobytwo']
labels=['ANT','CCT','DD','Motor','Stroop','DPX','Stop Signal','Two-by-Two']
assert not x.duplicated(['config_id','task','random_split']).any()
assert x.groupby(['config_id','task']).size().eq(10).all()
assert np.isfinite(x[['pred_probability_1_mean','pred_probability_1_std']]).all().all()
x['hidden_width']=x.hidden_dims.astype(str).str.split('-').str[0].astype(int)
x['rank_label']=x['rank'].fillna('Full').astype(str).str.replace('.0','',regex=False)
x['nodes_label']=x.hyper_hidden_dim.fillna('N/A').astype(str).str.replace('.0','',regex=False)
x['uniform_class']=x.pred_imbalance.eq(1)
x['near_constant_probability']=x.pred_probability_1_std.lt(1e-6)
parameters=[('num_hidden_layers','Hidden layers'),('hidden_width','Hidden units per layer'),('rank_label','Rank'),('participant_emb_dim','Embedding dimensions'),('mapping','Mapping'),('nodes_label','Hypernetwork nodes\n(nonlinear only)')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
records=[]
for metric, title in [('pred_probability_1_mean','Mean predicted P(y=1)'),('pred_probability_1_std','SD of predicted P(y=1)')]:
    fig,axes=plt.subplots(8,6,figsize=(16,18),sharey=True,layout='constrained')
    for col,(param,label) in enumerate(parameters):
        d=x if param!='nodes_label' else x[x.mapping.eq('nonlinear')]
        levels=sorted(d[param].unique(),key=lambda v:(not str(v).isdigit(),int(v) if str(v).isdigit() else str(v)))
        g=d.groupby(['task',param])[metric].mean()
        records.append(g.rename('value').reset_index().rename(columns={param:'level'}).assign(metric=metric,hyperparameter=param))
        for row,task in enumerate(tasks):
            ax=axes[row,col]; vals=g.loc[task].reindex(levels)
            ax.bar(range(len(levels)),vals,color='#39739D',width=.65)
            ax.set_xticks(range(len(levels)),[str(v) for v in levels],rotation=25 if param=='mapping' else 0)
            ax.set_ylim(0,1.08 if metric.endswith('mean') else .52)
            ax.spines[['top','right']].set_visible(False)
            ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
            if row==0:ax.set_title(label)
            if col==0:ax.set_ylabel(labels[row]+'\n'+('Mean probability' if metric.endswith('mean') else 'Probability SD'))
            for j,val in enumerate(vals):ax.text(j,val+.007,f'{val:.3f}',ha='center',fontsize=7)
    fig.suptitle(title+' by hyperparameter and task\nEach bar averages per-model/fold statistics; other hyperparameters pooled',fontsize=13)
    fig.savefig(out/f'{metric}_by_task.png',dpi=300)
    fig.savefig(out/f'{metric}_by_task.pdf')
    fig.savefig(out/f'{metric}_by_task.svg')
    plt.close(fig)
pd.concat(records).to_csv(out/'probability_hyperparameters_by_task.csv',index=False)
summary=x.groupby('task').agg(mean_probability=('pred_probability_1_mean','mean'),mean_probability_sd=('pred_probability_1_std','mean'),min_probability_sd=('pred_probability_1_std','min'),uniform_class_rate=('uniform_class','mean'),near_constant_probability_rate=('near_constant_probability','mean'),test_nll=('test_nll','mean'),test_accuracy=('test_accuracy','mean')).reindex(tasks)
# For uniform-class predictions, accuracy identifies the empirical test class rate.
u=x[x.uniform_class].copy()
u['test_rate_1']=np.where(u.pred_probability_1_mean.ge(.5),u.test_accuracy,1-u.test_accuracy)
check=u.groupby(['task','random_split']).test_rate_1.agg(['min','max'])
assert (check['max']-check['min']).abs().lt(1e-10).all()
rates=u.groupby(['task','random_split']).test_rate_1.first().reset_index()
p=rates.test_rate_1
assert p.gt(0).all() and p.lt(1).all()
rates['oracle_constant_nll']=-(p*np.log(p)+(1-p)*np.log1p(-p))
y=x.merge(rates,on=['task','random_split'],how='left',validate='many_to_one')
y['nll_gain_over_oracle_constant']=y.oracle_constant_nll-y.test_nll
gains=y.groupby('task').agg(oracle_constant_nll=('oracle_constant_nll','mean'),mean_nll_gain=('nll_gain_over_oracle_constant','mean'))
summary=summary.join(gains)
summary.to_csv(out/'probability_diagnostic_summary.csv')
selected='layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_8_reg_0.0'
sel=y[y.config_id.eq(selected)].groupby('task').agg(probability_mean=('pred_probability_1_mean','mean'),probability_sd=('pred_probability_1_std','mean'),uniform_class_rate=('uniform_class','mean'),test_nll=('test_nll','mean'),test_accuracy=('test_accuracy','mean'),oracle_constant_nll=('oracle_constant_nll','mean'),nll_gain=('nll_gain_over_oracle_constant','mean')).reindex(tasks)
sel.to_csv(out/'selected_probability_diagnostics.csv')
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,metric,title in zip(axes,['mean_probability','mean_probability_sd','uniform_class_rate'],['Mean P(y=1)','SD of P(y=1)','Uniform-class runs (%)']):
    vals=summary[metric]*(100 if metric=='uniform_class_rate' else 1)
    ax.barh(labels,vals,color='#39739D');ax.invert_yaxis();ax.set_title(title);ax.spines[['top','right']].set_visible(False)
    for i,v in enumerate(vals):ax.text(v,i,f' {v:.3f}' if metric!='uniform_class_rate' else f' {v:.1f}',va='center',fontsize=8)
    ax.set_xlim(0,1.15 if metric=='mean_probability' else .5 if metric=='mean_probability_sd' else 115)
fig.suptitle('New evaluation: constant class predictions do not imply constant probabilities')
fig.savefig(out/'probability_diagnostics_overview.png',dpi=300)
fig.savefig(out/'probability_diagnostics_overview.pdf')
fig.savefig(out/'probability_diagnostics_overview.svg')
plt.close(fig)
print(summary.to_string());print('\nSELECTED\n'+sel.to_string())
(out/'probability_diagnostics_notes.txt').write_text('All 320 configurations, eight tasks and ten folds included. Mean and SD pool valid trials within a run, then plots average runs equally. SD mixes between-participant and within-participant variation. No significance tests; folds are dependent. Oracle constant NLL uses test labels and is a diagnostic lower bound for a single constant probability, not an independently fitted baseline. Available only where uniform-class runs identify the test prevalence. Source CSVs and checkpoints unchanged.\n')
