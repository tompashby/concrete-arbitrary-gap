"""Rebuild ACL paper tables from immutable responses; never infer answers from traces."""
from pathlib import Path
import sys, json, hashlib, csv, re, math
from collections import Counter
import numpy as np
from scipy.stats import binomtest, norm
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from relational_experiment.model import Trial
from relational_experiment.runner import parse_reasoning_trial_answer, trial_fingerprint
OUT=ROOT/'results/common'; SRC=ROOT/'data/common'; FIG=ROOT/'generated/figures/common'
SEED=20260922; B=10000
MODELS={'qwen27':'Qwen3.8-27B','gemma26':'Gemma 4 26B-A4B','gemma31':'Gemma 4 31B','qwenmax':'Qwen3.8-Max'}
PREFIX={'qwen27':'qwen-local-plain-common-v2','gemma26':'gemma4-plain-common-v3','gemma31':'gemma4-31b-dense-v1','qwenmax':'dashscope-qwen38-max-plain-common-v1'}
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def csvout(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def texrows(name,rows):
    (OUT/name).write_text('\n'.join(r.rstrip('\\')+chr(92)*2 for r in rows)+'\n')
def ci(x):
    x=np.asarray(x,float); rng=np.random.default_rng(SEED); vals=[]
    for _ in range(B//500): vals.extend(x[rng.integers(0,len(x),(500,len(x)))].mean(1))
    return np.quantile(vals,[.025,.975]).tolist()
def paired(a,b):
    a=np.asarray(a,int);b=np.asarray(b,int); n10=int(sum((a==1)&(b==0)));n01=int(sum((a==0)&(b==1)))
    return dict(difference=float(np.mean(a-b)),ci=ci(a-b),left_only=n10,right_only=n01,p=binomtest(n10,n10+n01).pvalue if n10+n01 else 1.)
def holm(vals):
    order=np.argsort(vals);out=np.zeros(len(vals));r=0
    for j,i in enumerate(order):r=max(r,min(1,(len(vals)-j)*vals[i]));out[i]=r
    return out.tolist()
def features(t):
    t=t.lower()
    return {'translation':bool(re.search(r'\b(?:zav|pel|miv|tor|dax|wug|fep|nuk|lom)\b[^\n.]{0,80}(?:means|=|is|denotes|represents)[^\n.]{0,50}(?:parent|sibling|married|spouse|mother|father)',t)),
      'kinship':bool(re.search(r'\b(?:parent|child|mother|father|sibling|spouse|aunt|uncle|niece|nephew|cousin|married)\b',t)),
      'composition':bool(re.search(r'\b(?:therefore|thus|hence|so|implies|compose|combining)\b',t)),
      'direction':bool(re.search(r'\b(?:question asks|asked for|perspective|reverse|direction|relative to)\b',t)),
      'gender':bool(re.search(r'\b(?:male|female|gender)\b',t))}
def main():
    OUT.mkdir(parents=True, exist_ok=True);FIG.mkdir(parents=True, exist_ok=True)
    trials={r['trial_id']:Trial.from_dict(r) for r in read(SRC/'trials.jsonl')};probs={r['problem_id']:r for r in read(SRC/'problems.jsonl')};ids=sorted(probs)
    assert len(trials)==1000 and len(probs)==500
    assert sha(SRC/'trials.jsonl')=='a3c81f351212b85ef3f044654a8be90c5fcae3c8937f2688b37d266b13e9230b'
    assert sha(SRC/'problems.jsonl')=='fd0c1ec75182b130b42cd20269680d9324ac9225cffd360173aedb239f608681'
    manifest=[];data={};scores=[];summaries=[];changes=[];discrepancies=[];trace=[]
    for m,pre in PREFIX.items():
        paths=list(SRC.glob(f'responses-{pre}-depth-*.jsonl'))
        if m=='gemma26':paths += [SRC/'responses-gemma4-plain-common-v2-depth-off.jsonl']
        paths=[p for p in paths if not(m=='gemma31' and p.name.endswith('-off.jsonl'))]
        for p in sorted(paths):
            suffix=p.stem.split('-depth-')[1];seed=int(suffix.split('-seed')[1]) if '-seed' in suffix else 20260914
            lev=suffix.split('-seed')[0].replace('-repaired','');budget=0 if lev=='off' else int(lev)
            rr=read(p);assert len(rr)==1000 and len({r['trial_id'] for r in rr})==1000,p
            assert {r['trial_id'] for r in rr}==set(trials)
            manifest.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),rows=len(rr),model=m,budget=budget,seed=seed,configs=sorted({r['config_fingerprint'] for r in rr if r.get('config_fingerprint')}),trial_fingerprints_present=sum(bool(r.get('trial_fingerprint')) for r in rr)))
            cell={}
            for r in rr:
                t=trials[r['trial_id']]
                if r.get('trial_fingerprint'):assert r['trial_fingerprint']==trial_fingerprint(t),(p,r['trial_id'])
                assert (r.get('correct_letter') or r.get('correct_answer'))==t.correct_letter
                trunc=bool(r.get('truncated')) or r.get('finish_reason')=='length'
                final=r.get('final_answer_content') if 'final_answer_content' in r else r.get('final_content','')
                ans,mode,_,_,_=parse_reasoning_trial_answer(final or '',t,'',trunc)
                valid=ans is not None and not r.get('request_error');ok=valid and ans==t.correct_letter
                if bool(r.get('correct'))!=ok:discrepancies.append(dict(file=p.name,id=t.trial_id,old=r.get('correct'),new=ok,final=final))
                rt=r.get('reasoning_tokens');rt=rt if rt is not None else ((r.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
                row=dict(model=m,budget=budget,seed=seed,problem_id=t.problem_id,condition=t.condition,correct=int(ok),valid=int(valid),truncated=int(trunc),error=int(bool(r.get('request_error'))),answer=ans,gold=t.correct_letter,reasoning_tokens=rt,reasoning_words=len((r.get('reasoning_content') or '').split()),completion_tokens=r.get('completion_tokens') or (r.get('usage') or {}).get('completion_tokens'),recovered=int(valid and (final or '').strip().lower().rstrip('.! ')!=ans))
                scores.append(row);cell[(t.problem_id,t.condition)]=row
                if seed==20260914 and budget and t.condition=='arbitrary':trace.append(dict(**row,**features(r.get('reasoning_content') or '')))
            data[m,budget,seed]=cell
    dump('parser_discrepancies.json',discrepancies)
    for (m,b,s),cell in sorted(data.items()):
        for subset in ['full','nonfloor','great']:
            selected=[g for g in ids if subset=='full' or ((probs[g]['answer'] not in ['great-aunt','great-uncle'])==(subset=='nonfloor'))]
            c=[cell[g,'concrete']['correct'] for g in selected];a=[cell[g,'arbitrary']['correct'] for g in selected];pa=paired(c,a)
            row=dict(model=m,budget=b,seed=s,subset=subset,n=len(selected),concrete=np.mean(c),arbitrary=np.mean(a),gap=pa['difference'],gap_low=pa['ci'][0],gap_high=pa['ci'][1],p=pa['p'],concrete_only=pa['left_only'],arbitrary_only=pa['right_only'])
            for cond in ['concrete','arbitrary']:
                x=[cell[g,cond] for g in selected];l,h=ci([v['correct'] for v in x]);row.update({cond+'_low':l,cond+'_high':h,cond+'_valid':sum(v['valid'] for v in x),cond+'_conditional':sum(v['correct'] for v in x)/max(1,sum(v['valid'] for v in x)),cond+'_truncated':sum(v['truncated'] for v in x),cond+'_errors':sum(v['error'] for v in x)})
            summaries.append(row)
        if s==20260914 and b:
            for cond in ['concrete','arbitrary']:
                z=paired([cell[g,cond]['correct'] for g in ids],[data[m,0,s][g,cond]['correct'] for g in ids]);changes.append(dict(model=m,budget=b,condition=cond,**z))
    seedresults=[]; across=[]
    for s in [20260914,20260915,20260916]:
        for cond in ['concrete','arbitrary']:
            seedresults.append(dict(seed=s,condition=cond,**paired([data['gemma26',512,s][g,cond]['correct'] for g in ids],[data['qwenmax',512,s][g,cond]['correct'] for g in ids])))
    for r,p in zip(seedresults,holm([r['p'] for r in seedresults])):r['p_holm']=p
    for cond in ['concrete','arbitrary']:
        d=[np.mean([data['gemma26',512,s][g,cond]['correct']-data['qwenmax',512,s][g,cond]['correct'] for s in [20260914,20260915,20260916]]) for g in ids];across.append(dict(condition=cond,difference=np.mean(d),ci=ci(d)))
    model512=[]
    for cond in ['concrete','arbitrary']:
        model512.append(dict(condition=cond,**paired(
            [data['gemma31',512,20260914][g,cond]['correct'] for g in ids],
            [data['qwenmax',512,20260914][g,cond]['correct'] for g in ids])))
    combined=[np.mean([
        data['gemma31',512,20260914][g,cond]['correct']-
        data['qwenmax',512,20260914][g,cond]['correct']
        for cond in ['concrete','arbitrary']]) for g in ids]
    model512.append(dict(condition='graph mean across presentations',difference=float(np.mean(combined)),ci=ci(combined)))
    tokens=[]
    for (m,b,s),cell in sorted(data.items()):
        for cond in ['concrete','arbitrary']:
            x=[r for r in cell.values() if r['condition']==cond];v=[r['reasoning_tokens'] for r in x if r['reasoning_tokens'] is not None];w=[r['reasoning_words'] for r in x]
            tokens.append(dict(model=m,budget=b,seed=s,condition=cond,reported_n=len(v),token_quantiles=np.quantile(v,[.1,.25,.5,.75,.9]).tolist() if v else None,word_quantiles=np.quantile(w,[.1,.25,.5,.75,.9]).tolist()))
    fr=[];adjusted=[]
    for m in MODELS:
        tt=[r for r in trace if r['model']==m];levels=sorted({r['budget'] for r in tt})
        for f in ['translation','kinship','composition','direction','gender']:
            for b in levels:
                for y in [0,1]:
                    x=[r for r in tt if r['budget']==b and r['correct']==y];fr.append(dict(model=m,budget=b,outcome=y,feature=f,n=len(x),positive=sum(r[f] for r in x),rate=np.mean([r[f] for r in x]) if x else None))
            X=np.asarray([[1]+[int(r['budget']==b) for b in levels[1:]]+[math.log1p(r['reasoning_words']),int(r[f])] for r in tt],float);y=np.array([r['correct'] for r in tt]);
            fit=minimize(lambda z: np.logaddexp(0,X@z).sum()-y@(X@z),np.zeros(X.shape[1]),jac=lambda z:X.T@(expit(X@z)-y),method='BFGS',options={'maxiter':500})
            pr=expit(X@fit.x);h=X.T@((pr*(1-pr))[:,None]*X);inv=np.linalg.pinv(h);clusters=np.array([np.sum(X[[r['problem_id']==g for r in tt]]*(y-pr)[[r['problem_id']==g for r in tt],None],axis=0) for g in ids]);cov=inv@(clusters.T@clusters)@inv;se=math.sqrt(max(0,cov[-1,-1]));stable=np.linalg.matrix_rank(X)==X.shape[1] and max(abs(fit.x))<15 and se>0
            adjusted.append(dict(model=m,feature=f,log_odds=float(fit.x[-1]),se=se,ci=[float(fit.x[-1]-1.96*se),float(fit.x[-1]+1.96*se)],p=float(2*norm.sf(abs(fit.x[-1]/se))) if stable else None,interpretable=bool(stable),controls='budget categorical + log(1+trace words); graph-clustered sandwich SE'))
    csvout('scores.csv',scores);csvout('behavior.csv',summaries);csvout('trace_feature_rates.csv',fr)
    dump('changes.json',changes);dump('seed_comparison.json',dict(paired=seedresults,across=across));dump('model_512_comparison.json',model512);dump('tokens.json',tokens);dump('trace_adjusted.json',adjusted)
    dump('manifest.json',dict(dataset={p.name:sha(p) for p in [SRC/'trials.jsonl',SRC/'problems.jsonl']},sources=manifest,parser_sha256=sha(ROOT/'src/relational_experiment/runner.py'),analysis_sha256=sha(Path(__file__)),bootstrap_replicates=B,seed=SEED,score_discrepancies=discrepancies,total_responses=len(scores),environment=dict(python=sys.version,numpy=np.__version__)))
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(7,4.4),layout='constrained')
    for ax,m in zip(axs.flat,MODELS):
        rr=sorted([r for r in summaries if r['model']==m and r['seed']==20260914 and r['subset']=='full'],key=lambda r:r['budget']);xx=np.arange(len(rr))
        for cond,color,mark in [('concrete','#0072B2','o'),('arbitrary','#D55E00','s')]:
            v=np.array([r[cond] for r in rr])*100;lo=np.array([r[cond+'_low'] for r in rr])*100;hi=np.array([r[cond+'_high'] for r in rr])*100
            ax.errorbar(xx,v,yerr=[v-lo,hi-v],color=color,marker=mark,ms=3,lw=1.1,capsize=2,label=cond.capitalize())
        ax.set(title=MODELS[m],ylim=(0,104),xticks=xx,xticklabels=['Off' if not r['budget'] else str(r['budget']) for r in rr],ylabel='Accuracy (%)');ax.tick_params(axis='x',labelsize=7);ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(fontsize=8,loc='lower right');fig.supxlabel('Nominal reasoning budget (categorical spacing)',fontsize=9)
    for ext in ['pdf','png']:fig.savefig(FIG/f'accuracy.{ext}',dpi=300)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(7,4.4),layout='constrained')
    for ax,m in zip(axs.flat,MODELS):
        rr=sorted([r for r in summaries if r['model']==m and r['seed']==20260914 and r['subset']=='full'],key=lambda r:r['budget']);v=np.array([r['gap'] for r in rr])*100
        ax.errorbar(range(len(rr)),v,yerr=[v-np.array([r['gap_low'] for r in rr])*100,np.array([r['gap_high'] for r in rr])*100-v],fmt='o-',capsize=2,color='#009E73');ax.axhline(0,color='gray',lw=.6);ax.set(title=MODELS[m],xticks=range(len(rr)),xticklabels=['Off' if not r['budget'] else str(r['budget']) for r in rr],ylabel='C - A (points)');ax.tick_params(axis='x',labelsize=7)
    for ext in ['pdf','png']:fig.savefig(FIG/f'gaps.{ext}',dpi=300)
    rows=[]
    for m in MODELS:
        for b in [0,512]:
            r=next(r for r in summaries if r['model']==m and r['budget']==b and r['seed']==20260914 and r['subset']=='full');valid=r['concrete_valid']+r['arbitrary_valid']
            rows.append(f"{MODELS[m]} & {'Off' if not b else b} & {100*r['concrete']:.1f} & {100*r['arbitrary']:.1f} & {100*r['gap']:+.1f} & {valid} \\\\")
    texrows('primary_rows.tex',rows)
    fullrows=[]
    for r in summaries:
        if r['seed']!=20260914 or r['subset']!='full':continue
        fullrows.append(f"{MODELS[r['model']]} & {'Off' if not r['budget'] else r['budget']} & {100*r['concrete']:.1f} & {100*r['arbitrary']:.1f} & {100*r['gap']:+.1f} & [{100*r['gap_low']:.1f}, {100*r['gap_high']:.1f}] & {r['p']:.3g} \\\\")
    texrows('full_rows.tex',fullrows)
    print(json.dumps(dict(responses=len(scores),cells=len(data),primary=[r for r in summaries if r['subset']=='full' and r['seed']==20260914 and r['budget'] in [0,32,512]],seeds=across),indent=2))
if __name__=='__main__':main()
