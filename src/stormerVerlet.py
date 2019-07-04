import torch
from .iteration import iteration

def stormerVerlet(q,p,H,t,steps,rtol=1e-05,atol=1e-08,maxSteps=10,detach=True):
    Q = []
    P = []
    Hq,Hp = _force(H)
    p = p.requires_grad_()
    q = q.requires_grad_()
    for ss in range(steps):
        p_ = p
        p = iteration(p,lambda p:p_-t*0.5*Hq(q,p),rtol=rtol,atol=atol,maxSteps=maxSteps,detach=detach)

        q_ = q
        q = iteration(q,lambda q:q_+t*0.5*(Hp(q_,p)+Hp(q,p)),rtol=rtol,atol=atol,maxSteps=maxSteps,detach=detach)

        p = p-t*0.5*Hq(q,p)
        if detach:
            Q.append(q.detach())
            P.append(p.detach())
            p = p.detach().requires_grad_()
            q = q.detach().requires_grad_()
        else:
            Q.append(q)
            P.append(p)
    Q = torch.cat(Q,dim=0)
    P = torch.cat(P,dim=0)
    return Q,P

def _force(H):
    Hq = lambda q,p: torch.autograd.grad(H(q,p).sum(),q)[0]
    Hp = lambda q,p: torch.autograd.grad(H(q,p).sum(),p)[0]
    return Hq,Hp