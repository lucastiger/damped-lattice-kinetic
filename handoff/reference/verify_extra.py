import numpy as np, json, time, warnings; warnings.filterwarnings("ignore")
from fk import init_branch
from gen import Gen
t0=time.time(); out={}
# (e) kernel simplicity: smallest singular values of M0 and M0^T at c=0.89, N=2048
S,psi,sig=init_branch(200.0,2048)
for c in np.round(np.arange(0.881,0.8901,0.001),6): psi,sig,_,_=S.solve(c,psi,sig)
S.build(0.89); M0=S.jac_psi(psi)
sv=np.linalg.svd(M0,compute_uv=False); sv=np.sort(sv)
out['sv_M0_smallest3']=sv[:3].tolist()
print("(e) smallest singular values of M0 (N=2048,c=0.89):",sv[:3],"t=%.0f"%(time.time()-t0),flush=True)
# (f) refinement of the scope case mu2=-0.5 : N=2048 vs N=4096
for N in (2048,4096):
    G=Gen(L=200.0,N=N,gamma=0.1,mu=1.0,mu2=-0.5); p=np.zeros(N); f=0.5
    for c in [0.55,0.65,0.75,0.82]: p,f,r=G.solve(c,p,f)
    fp,kap,c1=G.scalars(p,0.82); rel=abs(kap+fp*c1)/abs(fp*c1)
    out['mu2m05_N%d'%N]=dict(f=f,fp=fp,kappa=kap,pred=-fp*c1,relerr=rel,res=r)
    print("(f) N=%d f=%.7f f'=%.7f kappa=%.7f pred=%.7f rel=%.2e res=%.1e t=%.0f"%(N,f,fp,kap,-fp*c1,rel,r,time.time()-t0),flush=True)
json.dump(out,open('verify_extra.json','w'),indent=1)
print("done")
