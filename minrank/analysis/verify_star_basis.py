"""Check the star-basis argument.
(1) Decoding identity: for 1 not in A', H[A',:] = eps * sum_{A in star, A disjoint A'} H[A,:]  (mod p)  iff  q | n-2k.
(2) rank_p H == alpha  iff  q | n-2k (direct ranks).
(3) the k+1 congruences C(N-1+t,k-1) = [t=k] + eps[t=0] (mod p), t=0..k  iff  q | N, for N up to 400."""
import itertools
from math import comb
import numpy as np
from flint import nmod_mat
def qpow(p,k):
    q=1
    while q<k: q*=p
    return q
P=[2,3,5,7,11,13]
# (3) pure arithmetic, wide range
for k in range(2,13):
    eps=(-1)**(k-1)
    for p in P:
        q=qpow(p,k)
        for N in range(1,401):
            ok=all((comb(N-1+t,k-1)-((t==k)+eps*(t==0)))%p==0 for t in range(k+1))
            assert ok==(N%q==0),(k,p,N)
print("[3] congruences <=> q | N : k<=12, p<=13, N<=400")
cnt=0
for k in range(2,6):
    for n in range(2*k+1,30):
        if comb(n,k)>2000: break
        S=list(itertools.combinations(range(n),k)); X=np.zeros((len(S),n),dtype=np.int64)
        for i,s in enumerate(S): X[i,list(s)]=1
        I=X@X.T; eps=(-1)**(k-1); H=np.eye(len(S),dtype=np.int64)+eps*(I==0)
        star=[i for i,s in enumerate(S) if 0 in s]; rest=[i for i,s in enumerate(S) if 0 not in s]
        Hs=H[star]
        for p in P:
            q=qpow(p,k); pred=(n-2*k)%q==0
            # (1) identity
            C=eps*(I[np.ix_(rest,star)]==0).astype(np.int64)
            ident=np.all((H[rest]-C@Hs)%p==0)
            r=nmod_mat(len(S),len(S),[int(v)%p for v in H.ravel()],p).rank()
            assert ident==pred==(r==comb(n-1,k-1)),(k,n,p,ident,pred,r)
            cnt+=1
print(f"[1,2] identity <=> rank=alpha <=> q|n-2k in {cnt} (n,k,p) cases")
