import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from HyperNetwork import HyperNN, BehavioralDataset, behavioral_collate_fn, preprocess_task, fit_model, test_model
from task_config import TASK_CONFIGS

class SharedNN(nn.Module):
    def __init__(self, inputs, participants, participant_bias=False):
        super().__init__()
        self.hidden=nn.Linear(inputs,8)
        self.output=nn.Linear(8,2)
        nn.init.xavier_uniform_(self.hidden.weight);nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.bias=nn.Embedding(participants,1) if participant_bias else None
        if self.bias is not None:nn.init.zeros_(self.bias.weight)
    def forward(self,x,participant_ids):
        h=torch.tanh(self.hidden(x));logits=self.output(h)
        if self.bias is not None:
            b=self.bias(participant_ids).unsqueeze(1)
            logits=logits+torch.cat([torch.zeros_like(b),b],dim=-1)
        return logits,h

if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path('Results/HyperNN_Grid_Search_8_Task');out=Path('Results/TwoByTwo_Bias_Comparison');out.mkdir(exist_ok=True)
    raw=pd.read_csv('Data/hypernetwork_data/two_by_two_data.csv')
    splits=pd.read_csv(root/'random_splits/twobytwo_random_splits.csv')
    raw=raw[raw.worker_id.isin(splits.worker_id.unique())].copy()
    data,xcols,ycol=preprocess_task(raw,TASK_CONFIGS['TwoByTwo'])
    data=data.merge(splits,on=['worker_id','participant_id','trial'],how='outer',validate='one_to_one',indicator=True)
    assert data._merge.eq('both').all()
    participants=data.participant_id.nunique()
    saved=pd.read_csv('Results/HyperNN_Grid_Search_Evaluation/test_results_all.csv')
    configs={'hyper_nonlinear':'layers_1_dims_8_rank_full_emb_4_nonlinear_nodes_8_reg_0.0',
             'hyper_linear':'layers_1_dims_8_rank_full_emb_4_linear_nodes_na_reg_0.0'}
    records=[];prediction_rows=[]
    for split in range(10):
        subsets={label:data[data[f'random_split_{split}'].eq(label)].sort_values(['participant_id','trial']).copy() for label in ['train','validation','test']}
        loaders={label:DataLoader(BehavioralDataset(d,xcols,ycol),batch_size=16,shuffle=False,collate_fn=behavioral_collate_fn) for label,d in subsets.items()}
        hold=subsets['test'];hold=hold[hold.choice_mask.eq(1)&hold[ycol].isin([0,1])].copy()
        # Save HyperNN predictions first and verify the forward implementation against saved NLL.
        models={}
        for label,config in configs.items():
            ck=torch.load(root/config/f'random_split_{split}/twobytwo_model.pt',map_location='cpu',weights_only=True)
            c=ck['extra_config']
            m=HyperNN(n_participants=participants,input_dim=len(xcols),hidden_dim=tuple(c['hidden_dims']),output_dim=2,participant_emb_dim=c['participant_emb_dim'],hyper_hidden_dim=c['hyper_hidden_dim'],rank=c['rank'],shared_right=c['shared_right'],mapping=c['mapping'],num_hidden_layers=c['num_hidden_layers'])
            m.load_state_dict(ck['model_state_dict']);metrics=test_model(m,loaders['test'],'cpu')
            expected=saved.loc[saved.config_id.eq(config)&saved.task.eq('twobytwo')&saved.random_split.eq(split),'test_nll'].item()
            assert abs(metrics['raw_loss']-expected)<1e-6,(label,split,metrics['raw_loss'],expected)
            models[label]=(m,ck['best_epoch'])
        for label in ['shared_nn','participant_bias_nn']:
            seed=49+10000*split
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            m=SharedNN(len(xcols),participants,participant_bias=label=='participant_bias_nn')
            max_epochs=1200 if label=='participant_bias_nn' else 400
            path=out/f'{label}_split_{split}_epochs_{max_epochs}.pt' if label=='participant_bias_nn' else out/f'{label}_split_{split}.pt'
            if label=='shared_nn' and path.exists():
                previous=torch.load(path,map_location='cpu',weights_only=True)
                if previous['history'][-1]['epoch']==400 and 400-previous['best_epoch']<50:
                    max_epochs=1200
                    path=out/f'{label}_split_{split}_epochs_1200.pt'
            if path.exists():
                ck=torch.load(path,map_location='cpu',weights_only=True);m.load_state_dict(ck['model_state_dict']);epoch=ck['best_epoch']
            else:
                train_loader=DataLoader(BehavioralDataset(subsets['train'],xcols,ycol),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed),collate_fn=behavioral_collate_fn)
                m,history,_=fit_model(m,train_loader,loaders['validation'],n_epochs=max_epochs,device='cpu',save_path=str(path),model_type='baseline',lr=.001,patience=50,min_delta=.0001,verbose=False,extra_config={'seed':seed,'hidden_dim':8,'participant_bias':label=='participant_bias_nn'})
                epoch=torch.load(path,map_location='cpu',weights_only=True)['best_epoch']
            models[label]=(m,epoch)
        for label,(m,epoch) in models.items():
            metrics=test_model(m,loaders['test'],'cpu')
            records.append(dict(model=label,split=split,test_nll=metrics['raw_loss'],accuracy=metrics['acc'],n_trials=metrics['n_trials'],best_epoch=epoch))
            pp=[]
            with torch.no_grad():
                for batch in loaders['test']:
                    logits,_=m(batch['x'],batch['participant_id']);mask=batch['mask'].bool()&batch['y'].ge(0)&batch['y'].lt(2)
                    pp.extend(logits.softmax(-1)[...,1][mask].tolist())
            assert len(pp)==len(hold)
            rows=hold[['worker_id','trial',ycol]].copy();rows['p']=pp;rows['model']=label;rows['split']=split
            prediction_rows.append(rows)
        for train_name,source in [('rate_8fold',subsets['train']),('rate_9fold',pd.concat([subsets['train'],subsets['validation']]))]:
            valid=source[source.choice_mask.eq(1)&source[ycol].isin([0,1])]
            counts=valid.groupby('worker_id')[ycol].agg(['sum','size']);rate=(counts['sum']+.5)/(counts['size']+1)
            p=hold.worker_id.map(rate).to_numpy();y=hold[ycol].to_numpy()
            nll=-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()
            records.append(dict(model=train_name,split=split,test_nll=nll,accuracy=((p>=.5)==y).mean(),n_trials=len(y),best_epoch=np.nan))
        pd.DataFrame(records).to_csv(out/'test_by_split.csv',index=False)
        print('Completed split',split,flush=True)
    results=pd.DataFrame(records)
    summary=results.groupby('model').agg(test_nll=('test_nll','mean'),accuracy=('accuracy','mean'),mean_best_epoch=('best_epoch','mean'))
    summary.to_csv(out/'summary.csv')
    predictions=pd.concat(prediction_rows,ignore_index=True)
    # Recover original conditions by the same participant-wise trial numbering.
    raw['trial']=raw.groupby('worker_id').cumcount()+1
    cols=[c for c in ['task_type','switch_type'] if c in raw]
    predictions=predictions.merge(raw[['worker_id','trial']+cols],on=['worker_id','trial'],validate='many_to_one')
    p=np.clip(predictions.p,1e-15,1-1e-15);y=predictions[ycol]
    predictions['nll']=-(y*np.log(p)+(1-y)*np.log1p(-p))
    predictions.to_csv(out/'test_predictions.csv',index=False)
    predictions.groupby(['model']+cols,dropna=False).agg(nll=('nll','mean'),mean_probability=('p','mean'),n_trials=('p','size')).to_csv(out/'condition_results.csv')
    print(summary.to_string(),flush=True)
