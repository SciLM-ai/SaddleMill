"""Deep quasi-Newton diagnostics and low-memory shifted L-BFGS solves.

Analysis helpers are force-call free.  Rigid-translation projection uses an
orthonormal three-vector Cartesian basis without materializing a dense projector.
The shifted solver follows the shifted L-BFGS recursion of Erway, Jain & Marcia
(Optimization Methods and Software 29, 992-1004, 2014; arXiv:1209.5141).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


def safe_norm(v):
    a=np.asarray(v,float).reshape(-1)
    if a.size==0:return 0.0
    m=float(np.max(np.abs(a)))
    if m==0:return 0.0
    if not np.isfinite(m):return float('inf')
    return m*float(np.linalg.norm(a/m))


def _validate_force(force):
    value=np.asarray(force,float).reshape(-1)
    if value.size==0 or not np.all(np.isfinite(value)):
        raise ValueError('force must be a nonempty finite vector')
    return value


def _validate_initial_hessian(value):
    alpha=float(value)
    if not np.isfinite(alpha) or alpha<=0.0:
        raise ValueError('initial_hessian must be finite and >0')
    return alpha


def _validate_shift(mu):
    value=float(mu)
    if not np.isfinite(value) or value<0.0:
        raise ValueError('mu must be finite and >=0')
    return value


def _validate_pairs(pairs, shape):
    validated=[]
    for index,item in enumerate(pairs):
        if len(item)<3:
            raise ValueError(f'pair {index} must contain s, y, and s_dot_y')
        s=np.asarray(item[0],float).reshape(-1)
        y=np.asarray(item[1],float).reshape(-1)
        sy=float(item[2])
        if s.shape!=shape or y.shape!=shape:
            raise ValueError(f'pair {index} shape mismatch')
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(y)) or not np.isfinite(sy):
            raise ValueError(f'pair {index} contains non-finite values')
        if abs(sy)<=np.finfo(float).tiny:
            raise ValueError(f'pair {index} has zero s_dot_y denominator')
        validated.append((s,y,sy,*item[3:]))
    return validated


def rigid_translation_basis(natoms:int)->np.ndarray:
    n=int(natoms)
    if n<1: raise ValueError('natoms must be >=1')
    Q=np.zeros((3*n,3),float)
    scale=1.0/np.sqrt(float(n))
    for a in range(n):
        Q[3*a+0,0]=scale; Q[3*a+1,1]=scale; Q[3*a+2,2]=scale
    return Q


def project_rigid(v,Q):
    x=np.asarray(v,float).reshape(-1)
    return x-Q@(Q.T@x)


def translation_ratio(v,Q,squared=False):
    x=np.asarray(v,float).reshape(-1); den=safe_norm(x)
    if den<=1e-300:return 0.0
    num=safe_norm(Q.T@x)
    r=num/den
    return float(r*r if squared else r)


def secant_row(s,y,Q=None):
    s=np.asarray(s,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
    ss=float(s@s); yy=float(y@y); sy=float(s@y)
    sn=safe_norm(s); yn=safe_norm(y)
    out={
      's_norm':sn,'y_norm':yn,'s_dot_y':sy,
      'secant_curvature': None if ss<=0 else sy/ss,
      'secant_cosine': None if sn*yn<=0 else sy/(sn*yn),
      'gamma_sy_over_yy': None if yy<=0 else sy/yy,
    }
    if Q is not None:
      out['s_translation_ratio']=translation_ratio(s,Q)
      out['y_translation_ratio']=translation_ratio(y,Q)
    return out


def progressive_lbfgs(force,pairs,initial_hessian,dynamic_h0,Q=None):
    f=np.asarray(force,float).reshape(-1)
    rows=[]; prev=None
    for m in range(len(pairs)+1):
        subset=pairs[:m]
        q=f.copy(); al=[]
        for s,y,sy,*_ in reversed(subset):
            s=np.asarray(s,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
            a=float(s@q)/float(sy); al.append(a); q-=a*y
        h0=1.0/float(initial_hessian)
        if dynamic_h0 and subset:
            s,y,sy,*_=subset[-1]; yy=float(np.asarray(y)@np.asarray(y))
            if yy>0 and sy>0 and np.isfinite(sy/yy): h0=float(sy/yy)
        p=h0*q
        for item,a in zip(subset,reversed(al)):
            s,y,sy,*_=item; s=np.asarray(s,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
            b=float(y@p)/float(sy); p+=s*(a-b)
        pn=safe_norm(p)
        maxatom=max((safe_norm(z) for z in p.reshape(-1,3)), default=0.0) if p.size%3==0 else None
        cos=None
        if prev is not None:
            den=safe_norm(prev)*pn
            if den>0: cos=float(prev@p/den)
        row={'history_depth':m,'prediction_norm':pn,'prediction_max_atom_norm':maxatom,
             'successive_cosine':cos,'h0_inverse_scale':h0}
        if Q is not None: row['prediction_translation_fraction']=translation_ratio(p,Q,squared=True)
        rows.append(row); prev=p.copy()
    return rows


def _effective_rank(svals,rtol=1e-10):
    if len(svals)==0:return 0
    cutoff=max(float(svals[0])*rtol,1e-15)
    return int(np.count_nonzero(np.asarray(svals)>cutoff))


def block_diagnostics(S,Y,Yraw=None,rtol=1e-10):
    S=np.asarray(S,float); Y=np.asarray(Y,float)
    if S.ndim!=2 or Y.shape!=S.shape: raise ValueError('S/Y shape mismatch')
    if S.shape[1]==0:return {}
    G=S.T@S; STY=S.T@Y; C=0.5*(STY+STY.T)
    u,svals,vt=np.linalg.svd(S,full_matrices=False)
    rank=_effective_rank(svals,rtol)
    curv=[]
    if rank:
        Ur=vt[:rank,:].T
        Gr=Ur.T@G@Ur; Cr=Ur.T@C@Ur
        ge,gv=np.linalg.eigh(Gr)
        keep=ge>max(float(np.max(ge))*rtol,1e-15)
        if np.any(keep):
            W=gv[:,keep]@np.diag(1.0/np.sqrt(ge[keep]))
            curv=np.linalg.eigvalsh(W.T@Cr@W).tolist()
    denom=max(float(np.linalg.norm(STY)),1e-30)
    out={'block_size':int(S.shape[1]),'effective_rank':rank,
         's_singular_values':svals.tolist(),
         'sts_condition': (None if rank==0 else float(svals[0]/svals[rank-1])),
         'asymmetry_relative':float(np.linalg.norm(STY-STY.T)/denom),
         'generalized_curvatures':curv,
         'generalized_curvature_min': (None if not curv else float(min(curv))),
         'generalized_curvature_max': (None if not curv else float(max(curv)))}
    if Yraw is not None:
        Yr=np.asarray(Yraw,float); M=S.T@Yr; Cr=0.5*(M+M.T)
        curvr=[]
        if rank:
            Ur=vt[:rank,:].T; Gr=Ur.T@G@Ur; Crr=Ur.T@Cr@Ur
            ge,gv=np.linalg.eigh(Gr); keep=ge>max(float(np.max(ge))*rtol,1e-15)
            if np.any(keep):
                W=gv[:,keep]@np.diag(1.0/np.sqrt(ge[keep])); curvr=np.linalg.eigvalsh(W.T@Crr@W).tolist()
        out.update({'raw_asymmetry_relative':float(np.linalg.norm(M-M.T)/max(float(np.linalg.norm(M)),1e-30)),
                    'raw_generalized_curvatures':curvr})
    return out


def _bfgs_direct_action(v,pairs,alpha):
    v=np.asarray(v,float).reshape(-1); out=float(alpha)*v
    ops=[]
    for s,y,sy,*_ in pairs:
        s=np.asarray(s,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
        bs=float(alpha)*s
        for so,yo,bso,sbso,syo in ops:
            bs-=bso*(float(bso@s)/sbso); bs+=yo*(float(yo@s)/syo)
        sbs=float(s@bs)
        if sbs<=0 or sy<=0: raise np.linalg.LinAlgError('nonpositive BFGS curvature')
        ops.append((s.copy(),y.copy(),bs.copy(),sbs,float(sy)))
    out=float(alpha)*v
    for s,y,bs,sbs,sy in ops:
        out-=bs*(float(bs@v)/sbs); out+=y*(float(y@v)/sy)
    return out


def _ordinary_lbfgs(force,pairs,initial_hessian,dynamic_h0=False):
    f=_validate_force(force); pairs=_validate_pairs(pairs,f.shape); _validate_initial_hessian(initial_hessian); q=f.copy(); al=[]
    for s,y,sy,*_ in reversed(pairs):
        s=np.asarray(s,float).reshape(-1); y=np.asarray(y,float).reshape(-1); a=float(s@q)/float(sy); al.append(a); q-=a*y
    h0=1.0/float(initial_hessian)
    if dynamic_h0 and pairs:
        s,y,sy,*_=pairs[-1]; yy=float(np.asarray(y)@np.asarray(y))
        if yy>0 and sy>0:h0=float(sy)/yy
    x=h0*q
    for item,a in zip(pairs,reversed(al)):
        s,y,sy,*_=item; s=np.asarray(s,float).reshape(-1); y=np.asarray(y,float).reshape(-1); b=float(y@x)/float(sy); x+=s*(a-b)
    return x

@dataclass(frozen=True)
class PreparedShiftedLBFGS:
    force: np.ndarray
    alpha: float
    rank_vectors: tuple[np.ndarray, ...]
    subspace_Q: np.ndarray
    subspace_eigenvectors: np.ndarray
    subspace_eigenvalues: np.ndarray
    force_subspace: np.ndarray
    force_residual: np.ndarray


def prepare_shifted_lbfgs(force,pairs,initial_hessian,dynamic_h0=False):
    """Precompute the history-dependent direct-BFGS rank vectors once.

    This is an implementation-only cache for repeated shifted solves at the
    same QN state.  It preserves the historical shifted recursion exactly;
    only the repeated reconstruction of BFGS rank vectors is removed.
    """
    r=_validate_force(force)
    pairs=_validate_pairs(pairs,r.shape)
    alpha=_validate_initial_hessian(initial_hessian)
    if dynamic_h0 and pairs:
        s,y,sy,*_=pairs[-1]; yy=float(np.asarray(y)@np.asarray(y))
        if yy>0 and sy>0: alpha=1.0/(float(sy)/yy)
    if not np.isfinite(alpha) or alpha<=0: raise np.linalg.LinAlgError('invalid base hessian')
    us=[]
    prior=[]
    for s,y,sy,*_ in pairs:
        s=np.asarray(s,float).reshape(-1); y=np.asarray(y,float).reshape(-1); sy=float(sy)
        bs=alpha*s.copy()
        for so,yo,bso,sbso,syo in prior:
            bs-=bso*(float(bso@s)/sbso); bs+=yo*(float(yo@s)/syo)
        sbs=float(s@bs)
        if sbs<=0 or sy<=0: raise np.linalg.LinAlgError('nonpositive curvature')
        prior.append((s,y,bs,sbs,sy))
        us.append(bs/np.sqrt(sbs)); us.append(y/np.sqrt(sy))
    rank_vectors=tuple(np.asarray(u,float).copy() for u in us)
    if rank_vectors:
        U=np.column_stack(rank_vectors)
        Q,R=np.linalg.qr(U,mode='reduced')
        signs=np.asarray([(-1.0)**(j+1) for j in range(U.shape[1])],float)
        Bsub=alpha*np.eye(Q.shape[1]) + (R*signs[np.newaxis,:])@R.T
        Bsub=0.5*(Bsub+Bsub.T)
        evals,evecs=np.linalg.eigh(Bsub)
        fq=Q.T@r
        fres=r-Q@fq
    else:
        Q=np.empty((r.size,0),float); evecs=np.empty((0,0),float); evals=np.empty(0,float)
        fq=np.empty(0,float); fres=r.copy()
    return PreparedShiftedLBFGS(r.copy(),alpha,rank_vectors,Q,evecs,evals,fq,fres)


def shifted_lbfgs_solve_prepared(prepared,mu):
    """Solve the historical shifted system through its compact BFGS subspace.

    The direct BFGS matrix is ``alpha I + U D U^T`` with alternating rank-one
    signs.  Only ``span(U)`` differs from ``alpha I``.  ``prepare`` diagonalizes
    that at-most-2m-dimensional subspace once; each trial shift then changes
    only scalar denominators.  No full 3N Hessian eigendecomposition is used.
    """
    mu=_validate_shift(mu); alpha=_validate_initial_hessian(prepared.alpha)
    force=_validate_force(prepared.force)
    if np.asarray(prepared.force_residual).reshape(-1).shape != force.shape:
        raise ValueError('prepared force residual shape mismatch')
    if alpha+mu<=0: raise np.linalg.LinAlgError('invalid base shift')
    evals=np.asarray(prepared.subspace_eigenvalues,float)
    if not np.all(np.isfinite(evals)):
        raise ValueError('prepared eigenvalues contain non-finite values')
    if evals.size and np.any(evals+mu <= 0.0):
        raise np.linalg.LinAlgError('shifted compact Hessian is not positive definite')
    x=np.asarray(prepared.force_residual,float)/(alpha+mu)
    if evals.size:
        coeff=np.asarray(prepared.subspace_eigenvectors,float).T @ np.asarray(prepared.force_subspace,float)
        coeff=coeff/(evals+mu)
        x=x + np.asarray(prepared.subspace_Q,float) @ (np.asarray(prepared.subspace_eigenvectors,float) @ coeff)
    return np.asarray(x,float)


def shifted_lbfgs_solve_reference(force,pairs,initial_hessian,mu,dynamic_h0=False):
    """Pre-v13 reference implementation retained for regression/microbench only."""
    r=_validate_force(force); pairs=_validate_pairs(pairs,r.shape); _validate_initial_hessian(initial_hessian); mu=_validate_shift(mu)
    if mu == 0.0: return _ordinary_lbfgs(r,pairs,initial_hessian,dynamic_h0)
    alpha=float(initial_hessian)
    if dynamic_h0 and pairs:
        s,y,sy,*_=pairs[-1]; yy=float(np.asarray(y)@np.asarray(y))
        if yy>0 and sy>0: alpha=1.0/(float(sy)/yy)
    if not np.isfinite(alpha) or alpha<=0 or alpha+mu<=0: raise np.linalg.LinAlgError('invalid base shift')
    us=[]; prior=[]
    for s,y,sy,*_ in pairs:
        s=np.asarray(s,float).reshape(-1); y=np.asarray(y,float).reshape(-1); sy=float(sy)
        bs=alpha*s.copy()
        for so,yo,bso,sbso,syo in prior:
            bs-=bso*(float(bso@s)/sbso); bs+=yo*(float(yo@s)/syo)
        sbs=float(s@bs)
        if sbs<=0 or sy<=0: raise np.linalg.LinAlgError('nonpositive curvature')
        prior.append((s,y,bs,sbs,sy))
        us.append(bs/np.sqrt(sbs)); us.append(y/np.sqrt(sy))
    base=1.0/(alpha+mu); x=base*r.copy(); ps=[]; taus=[]
    for j,u in enumerate(us):
        p=base*u.copy()
        for i,(pi,taui) in enumerate(zip(ps,taus)):
            p += ((-1.0)**i)*taui*float(pi@u)*pi
        denom=1.0+((-1.0)**(j+1))*float(p@u)
        if abs(denom)<1e-14 or not np.isfinite(denom): raise np.linalg.LinAlgError('shifted recursion denominator')
        tau=1.0/denom
        x += ((-1.0)**j)*tau*float(p@r)*p
        ps.append(p); taus.append(tau)
    return x


def shifted_lbfgs_solve(force,pairs,initial_hessian,mu,dynamic_h0=False):
    """Solve (B + mu I)p = force with the historical shifted recursion."""
    force=_validate_force(force); pairs=_validate_pairs(pairs,force.shape); _validate_initial_hessian(initial_hessian); mu=_validate_shift(mu)
    if mu == 0.0:
        return _ordinary_lbfgs(force,pairs,initial_hessian,dynamic_h0)
    prepared=prepare_shifted_lbfgs(force,pairs,initial_hessian,dynamic_h0)
    return shifted_lbfgs_solve_prepared(prepared,mu)

@dataclass
class ShiftResult:
    direction: np.ndarray
    mu: float
    iterations: int
    raw_norm: float
    regularized_norm: float
    fallback: str=''


def adaptive_shifted_lbfgs(force,pairs,initial_hessian,radius,dynamic_h0=False,tol=1e-8,maxiter=80):
    """Historical adaptive shifted-trust solve with cached QN preprocessing."""
    force=_validate_force(force); pairs=_validate_pairs(pairs,force.shape); _validate_initial_hessian(initial_hessian)
    radius=float(radius); tol=float(tol)
    if not np.isfinite(radius) or radius<=0: raise ValueError('radius must be finite and >0')
    if not np.isfinite(tol) or tol<=0: raise ValueError('tol must be finite and >0')
    if isinstance(maxiter,(bool,np.bool_)) or not isinstance(maxiter,(int,np.integer)) or int(maxiter)<1:
        raise ValueError('maxiter must be an integer >=1')
    maxiter=int(maxiter)
    raw=_ordinary_lbfgs(force,pairs,initial_hessian,dynamic_h0)
    rn=safe_norm(raw)
    if rn<=radius:return ShiftResult(raw,0.0,0,rn,rn)
    try:
        prepared=prepare_shifted_lbfgs(force,pairs,initial_hessian,dynamic_h0)
    except Exception as exc:
        return ShiftResult(raw,0.0,0,rn,rn,type(exc).__name__+':'+str(exc))
    lo=0.0; hi=max(float(initial_hessian),1.0)
    p=raw
    bracket_iterations=0
    for bracket_iterations in range(1,81):
        p=shifted_lbfgs_solve_prepared(prepared,hi)
        if safe_norm(p)<=radius:break
        hi*=2.0
    else:return ShiftResult(raw,0.0,80,rn,rn,'failed_to_bracket')
    it=0
    for it in range(1,maxiter+1):
        mid=0.5*(lo+hi); p=shifted_lbfgs_solve_prepared(prepared,mid); n=safe_norm(p)
        if abs(n-radius)<=tol*max(1.0,radius): return ShiftResult(p,mid,it,rn,n)
        if n>radius: lo=mid
        else: hi=mid
    p=shifted_lbfgs_solve_prepared(prepared,hi)
    return ShiftResult(p,hi,it,rn,safe_norm(p))

