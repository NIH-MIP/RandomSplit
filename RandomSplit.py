# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# January 2023
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import sys
import numpy as np
import scipy
import scipy.linalg.lapack as la
import torch
import torch.nn as nn
from pytorch_extras import geqp3, ormqr

class ObjectiveSplit(nn.Module):
    def __init__(self, W, p, rank=None, t = 1000):
        super().__init__()
        self.p = p
        self.t = t
        self.rank = min(W.shape[0], W.shape[1]) if rank is None else rank

        zeros = torch.zeros(self.rank, dtype=W.dtype)

        self.min_loss = -p*(1-p)

        h, jpvt, tau = geqp3(W.T)

        self.register_buffer("zeros", zeros)
        self.register_buffer("h", h)
        self.register_buffer("jpvt", jpvt)
        self.register_buffer("tau", tau)

    def Qv(self, v):
        #N, K = self.h.shape
        u = torch.cat((self.zeros, v), dim=0)[..., None]
        
        if v.device.type == "cpu":
            return ormqr("L", "N", a=self.h, tau=self.tau, c=u).squeeze(-1)
        else:
            # XXX: Bad! Real bad! Scipy and torch mixing! These are not perfectly compatible!!!
            return torch.ormqr(self.h, self.tau, u, left=True, transpose=False).squeeze(-1)

    def forward(self, v):
        p, t = self.p, self.t

        x = self.Qv(v)

        Zx = x - x.mean(dim=0, keepdim=True)

        a = p*t/torch.logsumexp(-t*Zx, dim=0, keepdim=True)
        b = (1-p)*t/torch.logsumexp(t*Zx, dim=0, keepdim=True)
        c = torch.cat((a,b), dim=0)
        c = -torch.logsumexp(-t*c, dim=0, keepdim=True)/t

        xxN = (Zx*c).var(dim=0, keepdim=True, correction=0)

        return -xxN
        
class ObjectiveFold(nn.Module):
    def __init__(self, W, F, rank=None, t = 1000):
        super().__init__()

        assert F > 1

        p = 1/F

        self.F = F
        self.p = p
        self.t = t

        self.min_loss = -(F-1)*p*(1-p) - (F-1)*(F-2)*p**2

        self.rank = min(W.shape[0], W.shape[1]) if rank is None else rank
        
        zeros = torch.zeros((self.rank,F-1), dtype=W.dtype)

        h, jpvt, tau = geqp3(W.T)

        self.register_buffer("zeros", zeros)
        self.register_buffer("h", h)
        self.register_buffer("jpvt", jpvt)
        self.register_buffer("tau", tau)

    def QV(self, V):
        V = V.view((-1, self.F-1))

        #N, K = self.h.shape
        U = torch.cat((self.zeros, V), dim=0)

        if V.device.type =="cpu":
            return ormqr("L", "N", a=self.h, tau=self.tau, c=U)
        else:
            # XXX: Bad! Real bad! Scipy and torch mixing! These are not perfectly compatible!!!
            return torch.ormqr(self.h, self.tau, U, left=True, transpose=False)

    def forward(self, V):
        p, t = self.p, self.t

        X = self.QV(V)

        ZX = X - X.mean(dim=0, keepdim=True)

        a = p*t/torch.logsumexp(-t*ZX, dim=0, keepdim=True)
        b = (1-p)*t/torch.logsumexp(t*ZX, dim=0, keepdim=True)
        c = torch.cat((a,b), dim=0)
        c = -torch.logsumexp(-t*c, dim=0, keepdim=True)/t

        XXN = (ZX*c).T.cov(correction=0)
        
        if XXN.ndim < 2:
            return -XXN
        else:
            return XXN.sum() - 2*XXN.trace()

class SolverSplit:
    def __init__(self, W, p, device="cpu", early_stop=0.8):
        K, N = W.shape

        self.W = W
        self.p = p
        self.rank=min(K, N)-1 # In case N <= K
        self.device = device
        self.early_stop = early_stop

        self.loss = ObjectiveSplit(torch.from_numpy(W), p, rank=self.rank).to(self.device)

    def f(self, v):
        v = torch.from_numpy(v).to(self.device)
        v.requires_grad=True

        y = self.loss(v)
        y.backward()

        y = y.squeeze(0).cpu().item()

        #print(f"loss = {y}, {self.loss.min_loss}")
        if y <= self.early_stop*self.loss.min_loss:
            g = np.zeros(v.shape)
            return y, g

        g = v.grad.cpu().numpy().copy()

        return y, g

    def compute_x(self, v):
        p = self.p

        with torch.no_grad():
            v = torch.from_numpy(v).to(self.device)
            x = self.loss.Qv(v)
            x -= x.mean() # Zx

            assert x.min() < 0 and x.max() > 0

            a = torch.minimum(-p/x.min(), (1-p)/x.max())
            x = a*x + p
            x = x.cpu().numpy().copy()

        return x

    def solve(self, v):
        method = "L-BFGS-B"

        options = {
            "ftol": 1e-5,
            "gtol": 1e-8,
            "maxiter": 10000,
            "disp": False,
        }

        fun = lambda x, obj : obj.f(x)

        res = scipy.optimize.minimize(fun=fun, x0=v, jac=True, args=(self,), method=method, options=options)

        return self.compute_x(res.x)

    def sample(self, rng):
        K, N = self.W.shape
        rank = self.rank

        for _ in range(5):
            v = rng.standard_normal(N-rank)
            v /= np.linalg.norm(v, ord=2)

            try:
                return self.solve(v)
            except AssertionError as e:
                print(f"Encountered an assertion: {e}")
                continue

        raise RuntimeError("Failed to find a solution.")

class SolverFold:
    def __init__(self, W, F, device="cpu", early_stop=0.8):
        K, N = W.shape

        self.W = W
        self.F = F
        self.p = 1/F
        self.rank=min(K, N)-1 # In case N <= K
        self.device=device
        self.early_stop = early_stop

        self.loss = ObjectiveFold(torch.from_numpy(W), F, rank=self.rank).to(self.device)

    def f(self, V):

        V = torch.from_numpy(V).to(self.device)
        V.requires_grad=True

        y = self.loss(V)
        y.backward()

        y = y.squeeze(0).cpu().item()

        #print(f"loss = {y}, {self.loss.min_loss}")
        if y <= self.early_stop*self.loss.min_loss:
            g = np.zeros(V.shape)
            return y, g

        g = V.grad.cpu().numpy().copy()

        return y, g
           
    def compute_X(self, V):
        p = self.p

        with torch.no_grad():
            V = torch.from_numpy(V).to(self.device)
            X = self.loss.QV(V)
            X -= X.mean(dim=0, keepdim=True) # Zx

            assert X.min() < 0 and X.max() > 0

            a = torch.minimum(-p/X.min(), (1-p)/X.max())
            X = a*X + p
            X = X.cpu().numpy().copy()

        return X

    def solve(self, V):
        method = "L-BFGS-B"

        options = {
            "ftol": 1e-5,
            "gtol": 1e-8,
            "maxiter": 10000,
        }

        fun = lambda x, obj : obj.f(x)

        res = scipy.optimize.minimize(fun=fun, x0=V, jac=True, args=(self,), method=method, options=options)

        #assert res.success

        return self.compute_X(res.x)

    def sample(self, rng):
        K, N = self.W.shape
        F = self.F
        rank = self.rank

        for _ in range(5):
            V = rng.standard_normal((N-rank, F-1))
            V /= np.linalg.norm(V, ord=2, axis=0)

            try:
                return self.solve(V.ravel())
            except AssertionError as e:
                print(f"Encountered an assertion: {e}")
                continue

        raise RuntimeError("Failed to find a solution.")

class NullSpaceSampler:
    def __init__(self, W, rank=None):
        self.rank = min(W.shape[0], W.shape[1]) if rank is None else rank
        #self.h, self.tau, work, info = la.dgeqrf(W.T)
        self.h, self.jpvt, self.tau, work, info = la.dgeqp3(W.T)

        assert info == 0

    def sample(self, rng, count=1):
        N, K = self.h.shape

        X = np.zeros((N,count))
        X[self.rank:, :] = rng.standard_normal((N-self.rank, count))

        # Query lwork first
        qc, work, info = la.dormqr(side="L", trans="N", a=self.h, tau=self.tau, c=X, lwork=-1)
        lwork = work[0]

        assert info == 0

        qc, work, info = la.dormqr(side="L", trans="N", a=self.h, tau=self.tau, c=X, lwork=lwork)

        assert info == 0

        return qc, X[self.rank:, :]

def RandomSplit(W, training_size, tries=10, random_state=None, max_batch_size=1000):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state
   
    N = W.shape[1]
    
    if training_size < 1:
        training_size = int(training_size*N)
        
    assert training_size >= 0 and training_size <= N
    
    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something
    
    if training_size == 0:
        return np.zeros(N, dtype=int), 0.0
        
    if training_size == N:
        return np.ones(N, dtype=int), 0.0
    
    # Remove rows with no counts over any instance
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    
    K = W.shape[0]
    
    #assert K > 1 and N >= K
    assert K > 1
    
    D = 1.0/D
    Z = np.eye(K) - 1.0/K
    
    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]
    
    # This is ZDW
    W = Z @ W

    if N < K:
        ind = np.arange(N)
        rng.shuffle(ind)

        x = np.zeros(N, dtype=int)
        x[ind[:training_size]] = 1

        res = np.linalg.norm(np.inner(W, x))

        return x, res
   

    sampler = NullSpaceSampler(W, rank=K-1)
    
    bestRes = -1.0
    bestX = None

    for i in range(0, tries, max_batch_size):
        batch_size = min(max_batch_size, tries - i)

        X, _ = sampler.sample(rng, count=batch_size)

        ind = np.argsort(X, axis=0)
        ind = ind[::-1, :]

        X = np.zeros((N, batch_size), dtype=int)
        np.put_along_axis(X, ind[:training_size, :], 1, axis=0)

        res = np.linalg.norm(W @ X, axis=0)

        b_min = np.argmin(res)

        if bestRes < 0.0 or res[b_min] < bestRes:
            bestRes = res[b_min]
            bestX = X[:, b_min].copy()

    return bestX, bestRes

def RandomSplitGlobal(W, training_size, random_state=None, device="cpu"):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state
   
    N = W.shape[1]
    
    if training_size < 1:
        p = training_size
        training_size = int(training_size*N)
    else:
        p = training_size / N
        
    assert training_size >= 0 and training_size <= N
    
    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something
    
    if training_size == 0:
        return np.zeros(N, dtype=int), 0.0
        
    if training_size == N:
        return np.ones(N, dtype=int), 0.0
    
    # Remove rows with no counts over any instance
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    
    K = W.shape[0]
    
    #assert K > 1 and N >= K
    assert K > 1

    D = 1.0/D
    Z = np.eye(K) - 1.0/K
    
    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]
    
    # This is ZDW
    W = Z @ W

    # Fat matrix, this reduces to randomly splitting
    # The null space are only those constant vectors and any kind of ranking/sorting of that 
    # is a random result
    if N < K:
        ind = np.arange(N)
        rng.shuffle(ind)

        x = np.zeros(N, dtype=int)
        x[ind[:training_size]] = 1

        res = np.linalg.norm(np.inner(W, x))

        return x, res

    solver = SolverSplit(W, p, device=device)

    #x = solver.sample(rng)
    
    # This was used in the paper. Find a good initial guess.
    sampler = NullSpaceSampler(W, rank=K-1)
    
    count = 1000
    X, V = sampler.sample(rng, count=count)
    
    ind = np.argsort(X, axis=0)
    ind = ind[::-1, :]
    
    X = np.zeros((N, count), dtype=int)
    np.put_along_axis(X, ind[:training_size, :], 1, axis=0)
    
    res = np.linalg.norm(W @ X, axis=0)
    
    b_min = np.argmin(res)
    
    #print(res[b_min])
    
    v = V[:, b_min]
    #print(v.shape)
    v = v/np.linalg.norm(v, ord=2)
    
    x = solver.solve(v)

    ind = np.argsort(x)
    ind = ind[::-1]
    x = np.zeros(N, dtype=int)
    x[ind[:training_size]] = 1

    res = np.linalg.norm(np.inner(W, x))

    return x, res

def BalancedCrossValidation(W, F, tries=10, aggregator=np.max, random_state=None, max_batch_size=1000):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]

    assert F > 1 and F <= N

    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something

    # Remove rows with no counts over any instance
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    
    K = W.shape[0]
    
    assert K > 1

    #lam = D/D.sum()

    D = 1.0/D
    Z = np.eye(K) - 1.0/K

    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]

    # This is ZDW
    W = Z @ W

    sampler = NullSpaceSampler(W, rank=min(K, N)-1)
    
    bestRes = None
    bestFolds = None

    for i in range(0, tries, max_batch_size):
        batch_size = min(max_batch_size, tries - i)

        X, _ = sampler.sample(rng, count=batch_size)

        ind = np.argsort(X, axis=0)
        ind = ind[::-1, :]

        res = []
        folds = []

        for f in range(F):
            val_begin = N*f//F
            val_end = N*(f+1)//F

            X = np.ones((N, batch_size), dtype=int)
            np.put_along_axis(X, ind[val_begin:val_end, :], 0, axis=0)

            folds.append(X)
            res.append(np.linalg.norm(W @ X, axis=0)[None, :])

        res = np.concatenate(tuple(res), axis=0)
        agg_res = aggregator(res, axis=0)
        b_min = np.argmin(agg_res)

        res = list(res[f,b_min].item() for f in range(F))
        folds = list(X[:,b_min].copy() for X in folds)

        if bestRes is None or agg_res[b_min] < aggregator(bestRes):
            bestRes = res
            bestFolds = folds

    return bestFolds, bestRes
    
def BalancedCrossValidationIncremental(W, F, tries=10, random_state=None):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]

    assert F > 1 and F <= N, f"F={F}, N={N}"

    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something

    # Remove rows with no counts over any instance
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    
    K = W.shape[0]
    
    #assert K > 1 and N >= K, f"K={K}, N={N}"
    assert K > 1

    #lam = D/D.sum()

    D = 1.0/D
    Z = np.eye(K) - 1.0/K

    Worig = W

    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]

    # This is ZDW
    W = Z @ W

    val_size = N//F
    training_size = N - val_size

    my_fold, my_res = RandomSplit(Worig, training_size, tries=tries, random_state=rng)

    index_map = np.argwhere(my_fold).squeeze(-1)
    #inv_index_map = { idx: i for i, idx in enumerate(index_map) }

    if F == 2:
        return [ 1-my_fold, my_fold ], [ my_res, my_res ]
    else:
        folds, _ = BalancedCrossValidationIncremental(Worig[:, my_fold != 0], F-1, tries=tries, random_state=rng)

    new_folds = []
    new_res = []

    for fold in folds:
        ind = np.asarray([ index_map[idx] for idx in np.argwhere(fold==0).squeeze(-1) ])
        fold = np.ones(N, dtype=int)
        fold[ind] = 0

        res = np.linalg.norm(np.inner(W, fold))

        new_folds.append(fold)
        new_res.append(res)


    new_folds.append(my_fold)
    new_res.append(my_res)

    return new_folds, new_res

def BalancedCrossValidationIncrementalGlobal(W, F, random_state=None, device="cpu"):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]

    assert F > 1 and F <= N, f"F={F}, N={N}"

    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something

    # Remove rows with no counts over any instance
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    
    K = W.shape[0]
    
    #assert K > 1 and N >= K, f"K={K}, N={N}"
    assert K > 1

    #lam = D/D.sum()

    D = 1.0/D
    Z = np.eye(K) - 1.0/K

    Worig = W

    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]

    # This is ZDW
    W = Z @ W

    val_size = N//F
    training_size = N - val_size

    my_fold, my_res = RandomSplitGlobal(Worig, training_size, random_state=rng, device=device)

    index_map = np.argwhere(my_fold).squeeze(-1)
    #inv_index_map = { idx: i for i, idx in enumerate(index_map) }

    if F == 2:
        return [ 1-my_fold, my_fold ], [ my_res, my_res ]
    else:
        folds, _ = BalancedCrossValidationIncrementalGlobal(Worig[:, my_fold != 0], F-1, random_state=rng, device=device)

    new_folds = []
    new_res = []

    for fold in folds:
        ind = np.asarray([ index_map[idx] for idx in np.argwhere(fold==0).squeeze(-1) ])
        fold = np.ones(N, dtype=int)
        fold[ind] = 0

        res = np.linalg.norm(np.inner(W, fold))

        new_folds.append(fold)
        new_res.append(res)


    new_folds.append(my_fold)
    new_res.append(my_res)

    return new_folds, new_res

def BalancedCrossValidationGlobal(W, F, random_state=None, device="cpu"):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]

    assert F > 1 and F <= N

    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something

    # Remove rows with no counts over any instance
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    
    K = W.shape[0]
    
    #assert K > 1 and N >= K, f"K={K}, N={N}"
    assert K > 1

    #lam = D/D.sum()

    D = 1.0/D
    Z = np.eye(K) - 1.0/K

    Worig = W

    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]

    # This is ZDW
    W = Z @ W

    solver = SolverFold(W, F, device=device)

    X = solver.sample(rng)

    ind = np.argsort(X, axis=0)

    scores_and_inds = [ [ (ind[n,f], f, X[ind[n,f],f]) for n in range(N) ] for f in range(F-1) ]
    scores_and_inds = sum(scores_and_inds, [])
    scores_and_inds = sorted(scores_and_inds, key=lambda tpl : tpl[2], reverse=True)

    folds = np.zeros((N,F), dtype=int)

    counts = np.zeros(N, dtype=int)
    fold_remaining = [ N*(f+1)//F - N*f//F for f in range(F-1) ]
    dup_count = 0

    for n, f, _ in scores_and_inds:
        if all(remaining <= 0 for remaining in fold_remaining):
            break

        if fold_remaining[f] <= 0:
            continue

        if counts[n] > 0:
            dup_count += 1
            continue

        folds[n,f] = 1
        counts[n] += 1
        fold_remaining[f] -= 1

    folds[:, -1] = 1-counts

    counts = folds.sum(axis=1)

    assert counts.min() == 1 and counts.max() == 1
                
    if dup_count > 0.05*N:
        print(f"Warning: Encountered more than 5% of duplicate indices ({dup_count*100/N:0.2f}%).")

    folds = [ 1-folds[:, f] for f in range(F) ]
    res = [ np.linalg.norm(W @ fold) for fold in folds ]

    return folds, res

def MakeRandomSplit(W, training_size, testing_size, column_map, tries=10, random_state=None):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]
    
    assert N <= len(column_map)
    
    if training_size < 1:
        training_size = int(training_size*N)
        
    if testing_size < 1:
        testing_size = int(testing_size*N)
        
    assert training_size >= 0 and testing_size >= 0 and training_size + testing_size <= N
    
    validation_size = N - (training_size + testing_size)
    
    xtest, restest = RandomSplit(W, testing_size, tries=tries, random_state=rng)
    xtrainval = 1-xtest
    
    testing_list = []
    
    for i in np.argwhere(xtest):
        cases = column_map[int(i)] # One column could represent a single patient with multiple scans!
        
        if not isinstance(cases, list):
            cases = [ cases ]
        
        testing_list += cases

    Wtrainval = W[:, np.argwhere(xtrainval).squeeze(-1)]
    
    column_map_trainval = []
    
    for i in np.argwhere(xtrainval):
        column_map_trainval.append(column_map[int(i)])
        
    xtrain, restrain = RandomSplit(Wtrainval, training_size, tries=tries, random_state=rng)
    
    validation_list = []
    training_list = []
    
    for i in range(len(xtrain)):
        cases = column_map_trainval[i]
        
        if not isinstance(cases, list):
            cases = [ cases ]
            
        if xtrain[i]:
            training_list += cases
        else:
            validation_list += cases
            
    return training_list, testing_list, validation_list, restrain, restest

def MakeRandomSplitGlobal(W, training_size, testing_size, column_map, random_state=None):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]
    
    assert N <= len(column_map)
    
    if training_size < 1:
        training_size = int(training_size*N)
        
    if testing_size < 1:
        testing_size = int(testing_size*N)
        
    assert training_size >= 0 and testing_size >= 0 and training_size + testing_size <= N
    
    validation_size = N - (training_size + testing_size)
    
    xtest, restest = RandomSplitGlobal(W, testing_size, random_state=rng)
    xtrainval = 1-xtest
    
    testing_list = []
    
    for i in np.argwhere(xtest):
        cases = column_map[int(i)] # One column could represent a single patient with multiple scans!
        
        if not isinstance(cases, list):
            cases = [ cases ]
        
        testing_list += cases

    Wtrainval = W[:, np.argwhere(xtrainval).squeeze(-1)]
    
    column_map_trainval = []
    
    for i in np.argwhere(xtrainval):
        column_map_trainval.append(column_map[int(i)])
        
    xtrain, restrain = RandomSplit(Wtrainval, training_size, tries=tries, random_state=rng)
    
    validation_list = []
    training_list = []
    
    for i in range(len(xtrain)):
        cases = column_map_trainval[i]
        
        if not isinstance(cases, list):
            cases = [ cases ]
            
        if xtrain[i]:
            training_list += cases
        else:
            validation_list += cases
            
    return training_list, testing_list, validation_list, restrain, restest

def MakeBalancedCrossValidation(W, F, column_map, testing_size=0, tries=10, aggregator=np.max, random_state=None):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]
    
    assert N <= len(column_map)

    if testing_size < 1:
        testing_size = int(testing_size*N)

    assert testing_size >= 0 and testing_size < N

    testing_list = []
    restest = 0.0

    if testing_size > 0:
        xtest, restest = RandomSplit(W, testing_size, tries=tries, random_state=rng)
        xcv = 1-xtest

        for i in np.argwhere(xtest):
            cases = column_map[int(i)] # One column could represent a single patient with multiple scans!
            
            if not isinstance(cases, list):
                cases = [ cases ]
            
            testing_list += cases

        Wcv = W[:, np.argwhere(xcv).squeeze(-1)]
        
        column_map_cv = []
    
        for i in np.argwhere(xcv):
            column_map_cv.append(column_map[int(i)])
    else:
        Wcv = W
        column_map_cv = column_map

    folds, res = BalancedCrossValidation(Wcv, F, tries=tries, aggregator=aggregator, random_state=rng)

    training_lists = []
    validation_lists = []

    for fold in folds:
        training_list = []
        validation_list = []

        for i in range(len(fold)):
            cases = column_map_cv[i]

            if not isinstance(cases, list):
                cases = [ cases ]

            if fold[i]:
                training_list += cases
            else:
                validation_list += cases                

        training_lists.append(training_list)
        validation_lists.append(validation_list)

    return training_lists, validation_lists, testing_list, res, restest

def MakeBalancedCrossValidationIncremental(W, F, column_map, testing_size=0, tries=10, random_state=None):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]
    
    assert N <= len(column_map)

    if testing_size < 1:
        testing_size = int(testing_size*N)

    assert testing_size >= 0 and testing_size < N

    testing_list = []
    restest = 0.0

    if testing_size > 0:
        xtest, restest = RandomSplit(W, testing_size, tries=tries, random_state=rng)
        xcv = 1-xtest

        for i in np.argwhere(xtest):
            cases = column_map[int(i)] # One column could represent a single patient with multiple scans!
            
            if not isinstance(cases, list):
                cases = [ cases ]
            
            testing_list += cases

        Wcv = W[:, np.argwhere(xcv).squeeze(-1)]
        
        column_map_cv = []
    
        for i in np.argwhere(xcv):
            column_map_cv.append(column_map[int(i)])
    else:
        Wcv = W
        column_map_cv = column_map

    folds, res = BalancedCrossValidationIncremental(Wcv, F, tries=tries, random_state=rng)

    training_lists = []
    validation_lists = []

    for fold in folds:
        training_list = []
        validation_list = []

        for i in range(len(fold)):
            cases = column_map_cv[i]

            if not isinstance(cases, list):
                cases = [ cases ]

            if fold[i]:
                training_list += cases
            else:
                validation_list += cases                

        training_lists.append(training_list)
        validation_lists.append(validation_list)

    return training_lists, validation_lists, testing_list, res, restest
    
def MakeBalancedCrossValidationGlobal(W, F, column_map, testing_size=0, random_state=None, device="cpu"):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]
    
    assert N <= len(column_map)

    if testing_size < 1:
        testing_size = int(testing_size*N)

    assert testing_size >= 0 and testing_size < N

    testing_list = []
    restest = 0.0

    if testing_size > 0:
        xtest, restest = RandomSplitGlobal(W, testing_size, random_state=rng, device=device)
        xcv = 1-xtest

        for i in np.argwhere(xtest):
            cases = column_map[int(i)] # One column could represent a single patient with multiple scans!
            
            if not isinstance(cases, list):
                cases = [ cases ]
            
            testing_list += cases

        Wcv = W[:, np.argwhere(xcv).squeeze(-1)]
        
        column_map_cv = []
    
        for i in np.argwhere(xcv):
            column_map_cv.append(column_map[int(i)])
    else:
        Wcv = W
        column_map_cv = column_map

    folds, res = BalancedCrossValidationGlobal(Wcv, F, random_state=rng, device=device)

    training_lists = []
    validation_lists = []

    for fold in folds:
        training_list = []
        validation_list = []

        for i in range(len(fold)):
            cases = column_map_cv[i]

            if not isinstance(cases, list):
                cases = [ cases ]

            if fold[i]:
                training_list += cases
            else:
                validation_list += cases                

        training_lists.append(training_list)
        validation_lists.append(validation_list)

    return training_lists, validation_lists, testing_list, res, restest

def MakeBalancedCrossValidationIncrementalGlobal(W, F, column_map, testing_size=0, random_state=None, device="cpu"):
    assert W.ndim == 2
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

    N = W.shape[1]
    
    assert N <= len(column_map)

    if testing_size < 1:
        testing_size = int(testing_size*N)

    assert testing_size >= 0 and testing_size < N

    testing_list = []
    restest = 0.0

    if testing_size > 0:
        xtest, restest = RandomSplitGlobal(W, testing_size, random_state=rng, device=device)
        xcv = 1-xtest

        for i in np.argwhere(xtest):
            cases = column_map[int(i)] # One column could represent a single patient with multiple scans!
            
            if not isinstance(cases, list):
                cases = [ cases ]
            
            testing_list += cases

        Wcv = W[:, np.argwhere(xcv).squeeze(-1)]
        
        column_map_cv = []
    
        for i in np.argwhere(xcv):
            column_map_cv.append(column_map[int(i)])
    else:
        Wcv = W
        column_map_cv = column_map

    folds, res = BalancedCrossValidationIncrementalGlobal(Wcv, F, random_state=rng, device=device)

    training_lists = []
    validation_lists = []

    for fold in folds:
        training_list = []
        validation_list = []

        for i in range(len(fold)):
            cases = column_map_cv[i]

            if not isinstance(cases, list):
                cases = [ cases ]

            if fold[i]:
                training_list += cases
            else:
                validation_list += cases                

        training_lists.append(training_list)
        validation_lists.append(validation_list)

    return training_lists, validation_lists, testing_list, res, restest
