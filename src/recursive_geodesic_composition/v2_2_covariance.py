"""V2.2 TRAIN-only covariance objectives; no corpus or cache imports."""
import torch

def _cov(x, eps=1e-6):
 if x.ndim!=2 or x.shape[0]<4: raise ValueError('covariance requires >=4 rows')
 if not torch.isfinite(x).all(): raise ValueError('nonfinite embedding')
 z=x-x.mean(0,keepdim=True); return z.T@z/(x.shape[0]-1)
def covariance_frobenius(g,t,eps=1e-6):
 """Dimension-normalized squared covariance difference."""
 d=g.shape[1]; return ((_cov(g,eps)-_cov(t,eps)).square().mean())
def whitened_covariance(g,t,ridge=1e-4):
 """Teacher minibatch whitening; ridge is trace/d scaled and detached only for scale."""
 ct=_cov(t);scale=ct.diagonal().mean().clamp_min(1e-8); eye=torch.eye(ct.shape[0],device=t.device,dtype=t.dtype)
 l=torch.linalg.cholesky(ct+ridge*scale*eye); wg=torch.linalg.solve_triangular(l,(g-g.mean(0)).T,upper=False).T; wt=torch.linalg.solve_triangular(l,(t-t.mean(0)).T,upper=False).T
 return covariance_frobenius(wg,wt),torch.linalg.cond(ct+ridge*scale*eye)
def balanced(losses):
 """Equal corpus contribution; caller supplies only valid corpus losses."""
 if len(losses)!=4: raise ValueError('requires four corpus terms')
 return torch.stack(losses).mean()
