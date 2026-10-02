# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# September 2026
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

import numpy as np
from RandomSplit import BalancedCrossValidationIncrementalGlobal

def PureRandomCrossValidation(W, F, tries=10, aggregator=np.max, random_state = None, max_batch_size=1000):
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
    
    assert K > 1 and N >= K
    
    D = 1.0/D
    Z = np.eye(K) - 1.0/K
    
    # This is the same as D*W... just in numpy weirdness
    W = W*D[..., None]
    
    # This is ZDW
    W = Z @ W

    bestRes = None
    bestFolds = None

    max_batch_size = min(max_batch_size, tries)
    
    ind = np.repeat(np.arange(N)[:, None], max_batch_size, axis=1)

    for i in range(0, tries, max_batch_size):
        batch_size = min(max_batch_size, tries - i)
        ind = ind[:, :batch_size]

        ind = rng.permuted(ind, axis=0)

        res = []
        folds = []

        for f in range(F):
            val_begin = N*f//F
            val_end = N*(f+1)//F

            X = np.ones((N, batch_size))
            #X[ind[val_begin:val_end], :] = 0 # Mask out validation set
            np.put_along_axis(X, ind[val_begin:val_end, :], 0, axis=0)

            folds.append(X)
            res.append(np.linalg.norm(W @ X, axis=0)[None, ...])

        res = np.concatenate(tuple(res), axis=0)
        agg_res = aggregator(res, axis=0)
        b_min = np.argmin(agg_res)

        res = list(res[f,b_min].item() for f in range(F))
        folds = list(X[:,b_min].copy() for X in folds)

        if bestRes is None or agg_res[b_min] < aggregator(bestRes):
            bestRes = res
            bestFolds = folds
            
    return bestFolds, bestRes

def RunBenchmark():
    K = 11
    N = 200
    F = 5
    numRuns=100
    tries=1000
    aggregator=np.max
    
    np.random.seed(727)
    seeds = np.random.randint(size=numRuns, low=1, high=2**31-1)

    allRes = np.zeros(numRuns)
    allResRandom = np.zeros(numRuns)

    ind = np.arange(N)
    
    M = int(0.9*N)
    
    for i in range(numRuns):
        np.random.seed(seeds[i])
        W = np.random.randint(size=[K,N], low=1, high=10)
        
        W[0, :] = 1
        W[7, :] = 0 # Test zero row removal support
        
        for k in range(1,K):
            np.random.shuffle(ind)
            W[k, :][ind[:M]] = 0
        
        assert np.all(W.max(axis=0) > 0)
        
        #expected = np.round(((F-1.0)/F)*W.sum(axis=1)).astype(int)
        
        folds, res = BalancedCrossValidationIncrementalGlobal(W, F, random_state=seeds[i])

        #for f, fold in enumerate(folds):
        #    svd = np.inner(W, fold)
        #    print(f"SVD {f}: {svd}")

        #print(f"Expected: {expected}\n")

        allRes[i] = aggregator(res)

        folds, res = PureRandomCrossValidation(W, F, tries=tries, aggregator=aggregator, random_state=seeds[i])

        #for f, fold in enumerate(folds):
        #    random = np.inner(W, fold)
        #    print(f"Random {f}: {random}")

        #print(f"Expected: {expected}\n")

        allResRandom[i] = aggregator(res)

    print(f"QR: {allRes.mean()} +/- {allRes.std()}")
    print(f"Random: {allResRandom.mean()} +/- {allResRandom.std()}")

if __name__ == "__main__":
    RunBenchmark()

