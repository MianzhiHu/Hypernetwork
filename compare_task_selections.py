import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from cca_zoo.linear import CCA, GCCA
from CKA.CKA import CKA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('Results/HyperNN_Grid_Search_Evaluation')
out=Path('Results/Task_Selection_Comparison');out.mkdir(exist_ok=True)
figdir=Path('Figures/Task_Selection_Comparison');figdir.mkdir(exist_ok=True)
tasks=['cct','dd','motor','stopsignal','dpx','twobytwo'];four=tasks[:4]
selected='layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_8_reg_0.0'
test=pd.read_csv(root/'test_results_all.csv');test=test[test.task.isin(tasks)].copy()
configs=sorted(p.name.removesuffix('_cct.npz') for p in (root/'Embedding_Stability/Aligned_Embeddings').glob('*_cct.npz'))
test.to_csv(out/'test_results_snapshot_partial.csv', index=False)
completed_summary=pd.read_csv(root/'test_results_summary.csv')
completed_summary.to_csv(out/'completed_test_summary_snapshot.csv',index=False)
ids=pd.read_csv(root/'Embedding_Stability/participant_map.csv').sort_values('participant_id')
workers=ids.worker_id.tolist()
prevalence={};baselines=[]
files={'stopsignal':'stop_signal','twobytwo':'two_by_two'}
for task in tasks:
 d=pd.read_csv(Path('Data/hypernetwork_data')/f'{files.get(task,task)}_data.csv')
 d['trial']=d.groupby('worker_id').cumcount()+1
 if task=='dd':d['y']=d.choice.astype('category').cat.codes-1
 elif task=='cct':d['y']=d.action.astype('category').cat.codes
 elif task=='dpx':d['y']=d.correct.astype('category').cat.codes
 else:d['y']=d.correct.astype(int)
 valid=d.y.isin([0,1])
 if task not in ['motor','stopsignal']:valid &= d.responded.eq(1)
 saved=pd.read_csv(Path('Results/HyperNN_Grid_Search/random_splits')/f'{task}_random_splits.csv')
 d=d.merge(saved,on=['worker_id','trial'],how='left',validate='one_to_one')
 assert d.fold.notna().all()
 d=d.loc[valid.to_numpy()].copy()
 prevalence[task]=d.groupby('worker_id').y.mean().reindex(workers).to_numpy()
 for split in range(10):
  train=d[d[f'random_split_{split}'].ne('test')];hold=d[d[f'random_split_{split}'].eq('test')]
  counts=train.groupby('worker_id').y.agg(['sum','size'])
  rates=(counts['sum']+.5)/(counts['size']+1) # Jeffreys smoothing, avoiding log(0).
  p=hold.worker_id.map(rates).to_numpy();y=hold.y.to_numpy()
  nll=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()
  baselines.append(dict(task=task,random_split=split,baseline_nll=nll,n_trials=len(hold)))
baselines=pd.DataFrame(baselines);baselines.to_csv(out/'response_rate_baselines.csv',index=False)
test=test.merge(baselines,on=['task','random_split'],validate='many_to_one')
assert test.n_test_trials.eq(test.n_trials).all()
test['nll_gain']=test.baseline_nll-test.test_nll
test['uniform_class']=test.pred_imbalance.eq(1)
test.groupby('task').agg(test_nll=('test_nll','mean'),baseline_nll=('baseline_nll','mean'),nll_gain=('nll_gain','mean'),uniform_class=('uniform_class','mean'),probability_sd=('pred_probability_1_std','mean')).to_csv(out/'task_prediction_summary.csv')

pair_records=[];joint_records=[];stability=[];depth_records=[];prevalence_records=[];heldout=[]
cache={};cka=CKA();folds=list(KFold(5,shuffle=True,random_state=42).split(workers))
for ci,config in enumerate(configs):
 views={};svds={}
 for task in tasks:
  with np.load(root/'Embedding_Stability/Aligned_Embeddings'/f'{config}_{task}.npz',allow_pickle=False) as z:
   assert list(z['worker_id'].astype(str))==workers
   s=z['scores'];v=z['svd_scores']
  s=np.stack([StandardScaler().fit_transform(a) for a in s])
  v=np.stack([StandardScaler().fit_transform(a) for a in v])
  views[task]=s.mean(0);svds[task]=v.mean(0)
  fold_r=np.corrcoef(s[:,:,0])[np.triu_indices(10,1)].mean()
  stability.append(dict(config_id=config,task=task,first_component_r=fold_r))
  # Descriptive fraction of consensus variance linearly associated with own response rate.
  x=views[task]-views[task].mean(0);design=np.column_stack([np.ones(len(workers)),prevalence[task]])
  fitted=design@np.linalg.lstsq(design,x,rcond=None)[0]
  prevalence_records.append(dict(config_id=config,task=task,response_rate_variance_fraction=np.sum(fitted**2)/np.sum(x**2)))
 cache[config]=views
 for a,b in combinations(tasks,2):
  x,y=views[a],views[b]
  r=float(CCA(latent_dimensions=1).fit([x,y]).score([x,y])[0])
  sx,sy=svds[a],svds[b]
  sr=float(CCA(latent_dimensions=1).fit([sx,sy]).score([sx,sy])[0])
  design=np.column_stack([np.ones(len(workers)),prevalence[a],prevalence[b]])
  residual=[z-design@np.linalg.lstsq(design,z,rcond=None)[0] for z in [x,y]]
  rr=float(CCA(latent_dimensions=1).fit(residual).score(residual)[0])
  pair_records.append(dict(config_id=config,task_a=a,task_b=b,pair=a+'__'+b,r=r,svd_r=sr,residual_r=rr))
  if config==selected:
   for fi,(train,hold) in enumerate(folds):
    m=CCA(latent_dimensions=1).fit([x[train],y[train]])
    z=m.transform([x[hold],y[hold]])
    heldout.append(dict(pair=a+'__'+b,fold=fi,r=np.corrcoef(z[0][:,0],z[1][:,0])[0,1]))
 for nt,ts in [(4,four),(6,tasks)]:
  for rep,vv in [('raw',views),('svd',svds)]:
   v=[StandardScaler().fit_transform(vv[t]) for t in ts]
   m=GCCA(latent_dimensions=1).fit(v)
   joint_records.append(dict(config_id=config,n_tasks=nt,representation=rep,gcca_r=float(m.score(v)[0])))
 if ci%40==0:print('Processed',ci+1,'/',len(configs),flush=True)
pairs=pd.DataFrame(pair_records);joint=pd.DataFrame(joint_records);stab=pd.DataFrame(stability)
pairs.to_csv(out/'pairwise_results.csv',index=False);joint.to_csv(out/'joint_results.csv',index=False);stab.to_csv(out/'stability.csv',index=False)
pd.DataFrame(prevalence_records).to_csv(out/'response_rate_embedding_variance.csv',index=False)
pd.DataFrame(heldout).to_csv(out/'selected_participant_heldout_cca.csv',index=False)
# Reproduce current four-task values to ensure identical alignment/CCA pipeline.
old=pd.read_csv(root/'Cross_Task_Convergence/cca_by_task_pair.csv')
check=pairs.merge(old,on=['config_id','task_a','task_b'])
assert np.allclose(check.r,check.first_cca,atol=1e-8)

summary=[];typical=[];pair_summaries=[]
for nt,ts in [(4,four),(6,tasks)]:
 for depth in ['all','one_layer']:
  keep=[c for c in configs if depth=='all' or c.startswith('layers_1_')]
  p=pairs[pairs.config_id.isin(keep)&pairs.task_a.isin(ts)&pairs.task_b.isin(ts)]
  mat=p.pivot(index='config_id',columns='pair',values='r')
  corr=np.corrcoef(mat);upper=corr[np.triu_indices(len(mat),1)]
  np.fill_diagonal(corr,np.nan)
  typicality=np.tanh(np.nanmean(np.arctanh(np.clip(corr,-1+1e-12,1-1e-12)),axis=1))
  tr=pd.DataFrame(dict(config_id=mat.index,typicality_r=typicality,n_tasks=nt,depth=depth));typical.append(tr)
  t=test[test.config_id.isin(keep)&test.task.isin(ts)]
  st=stab[stab.config_id.isin(keep)&stab.task.isin(ts)]
  j=joint[joint.config_id.isin(keep)&joint.n_tasks.eq(nt)&joint.representation.eq('raw')]
  summary.append(dict(n_tasks=nt,depth=depth,n_configs=len(keep),mean_pair_r=p.r.mean(),mean_svd_pair_r=p.svd_r.mean(),mean_residual_r=p.residual_r.mean(),joint_gcca=j.gcca_r.mean(),stability_r=st.first_component_r.mean(),pattern_agreement=np.tanh(np.arctanh(np.clip(upper,-1+1e-12,1-1e-12)).mean()),median_pattern_agreement=np.median(upper),nll_gain_partial_snapshot=t.nll_gain.mean(),uniform_class_rate_partial_snapshot=t.uniform_class.mean(),n_performance_rows=len(t),selected_typicality=tr.set_index('config_id').loc[selected,'typicality_r']))
  ps=p.groupby('pair').r.agg(['mean','std']).reset_index().assign(n_tasks=nt,depth=depth);pair_summaries.append(ps)
summary=pd.DataFrame(summary);summary.to_csv(out/'four_way_summary.csv',index=False)
pd.concat(typical).to_csv(out/'typicality.csv',index=False);pd.concat(pair_summaries).to_csv(out/'pair_summaries.csv',index=False)
for c in configs:
 if not c.startswith('layers_1_'):continue
 # Change layer count and repeated width only; match all other hyperparameters.
 width=c.split('_')[3];other=c.replace('layers_1_dims_'+width+'_','layers_2_dims_'+width+'x'+width+'_',1)
 if other not in cache:continue
 for t in tasks:
  a,b=cache[c][t],cache[other][t]
  depth_records.append(dict(config_1=c,config_2=other,task=t,cca_r=float(CCA(latent_dimensions=1).fit([a,b]).score([a,b])[0]),cka=cka.linear_CKA(a,b)))
depth=pd.DataFrame(depth_records);depth.to_csv(out/'matched_depth_embeddings.csv',index=False)
print('Matched depth records',len(depth),flush=True)
# Matched association vectors and performance differences.
matched=[]
for c,other in depth[['config_1','config_2']].drop_duplicates().itertuples(index=False,name=None):
 for nt,ts in [(4,four),(6,tasks)]:
  z=pairs[pairs.task_a.isin(ts)&pairs.task_b.isin(ts)].pivot(index='config_id',columns='pair',values='r')
  a,b=z.loc[c].to_numpy(),z.loc[other].to_numpy()
  t=completed_summary[completed_summary.task.isin(four)].groupby('config_id').test_nll.mean()
  matched.append(dict(config_1=c,n_tasks=nt,pattern_r=np.corrcoef(a,b)[0,1],mean_cca_change=(b-a).mean(),nll_change=t.get(other,np.nan)-t.get(c,np.nan)))
pd.DataFrame(matched).to_csv(out/'matched_depth_patterns.csv',index=False)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,4,figsize=(14,4.5),layout='constrained')
for ax,row in zip(axes,summary.itertuples()):
 ts=four if row.n_tasks==4 else tasks
 p=pd.concat(pair_summaries);p=p[p.n_tasks.eq(row.n_tasks)&p.depth.eq(row.depth)]
 m=np.eye(len(ts))
 for r in p.itertuples():
  a,b=r.pair.split('__');i,j=ts.index(a),ts.index(b);m[i,j]=m[j,i]=r.mean
 im=ax.imshow(m,vmin=0,vmax=1,cmap='Blues')
 for i in range(len(ts)):
  for j in range(len(ts)):
   if i!=j:ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if m[i,j]>.65 else 'black')
 ax.set_xticks(range(len(ts)),ts,rotation=65,ha='right');ax.set_yticks(range(len(ts)),ts)
 ax.set_title(f'{row.n_tasks} tasks | '+('1 + 2 layers' if row.depth=='all' else '1 layer')+f'\nPattern agreement r={row.pattern_agreement:.2f}')
fig.colorbar(im,ax=axes,label='Mean first CCA r',shrink=.6)
fig.savefig(figdir/'four_way_comparison.png',dpi=300);fig.savefig(figdir/'four_way_comparison.pdf');fig.savefig(figdir/'four_way_comparison.svg')
print(summary.to_string(index=False),flush=True)
print('DEPTH\n',depth.groupby('task')[['cca_r','cka']].mean().to_string(),flush=True)
