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
#import sklearn.metrics.hamming_loss
from scipy.optimize import linear_sum_assignment

# With assistance from duck.ai (GPT 5.4 Nano)
def min_avg_hamming(A, B):
    # A, B: (10, N)
    C = np.count_nonzero(A[:, None, :] != B[None, :, :], axis=-1)  # (10,10)
    r, c = linear_sum_assignment(C)
    return C[r, c].mean()

def variability_over_all_pairs(X):
    # X: (M, 10, N)
    X = np.asarray(X)
    M = X.shape[0]

    print(X.shape)
    
    total = 0.0
    count = 0

    for g in range(M - 1):
        for h in range(g + 1, M):
            total += min_avg_hamming(X[g], X[h])
            count += 1

    return total / count
# End AI contribution

def labels_distribution(Y, folds, validation=True, lam=0):
    D = Y.sum(axis=0)

    Y = Y[:, D > 0]
    D = D[D > 0]

    Q = Y.shape[1]
    N = Y.shape[0]

    D += lam
    N = N + 2*lam

    ld = 0

    for fold in folds:
        if validation:
            fold = np.logical_not(fold)
        else:
            fold = (fold != 0)

        Dj = Y[fold, :].sum(axis=0) + lam
        Nj = fold.sum() + 2*lam
        assert np.all(Nj >= Dj)

        NjmDj = Nj - Dj
        if Nj > 0 and np.any(NjmDj == 0):
            #NjmDj[NjmDj == 0] = Nj
            ld += np.inf
            continue
        else: # 0/0 = 0?
            NjmDj[NjmDj == 0] = 1

        ld += np.abs(Dj/NjmDj - D/(N - D)).sum()
    
    ld /= (len(folds)*Q)

    return ld

def examples_distribution(Y, folds, validation=True):
    D = Y.sum(axis=0)

    Y = Y[:, D > 0]
    D = D[D > 0]

    N = Y.shape[0]

    F = len(folds)

    cj = N/len(folds)

    ed = 0
    for fold in folds:
        if validation:
            fold = np.logical_not(fold)
        else:
            fold (fold != 0)

        ed += np.abs(fold.sum() - cj)

    ed /= len(folds)

    return ed

def fz(Y, folds, validation=True):
    D = Y.sum(axis=0)

    Y = Y[:, D > 0]
    D = D[D > 0]

    count = 0
    for fold in folds:
        if validation:
            fold = np.logical_not(fold)
        else:
            fold = (fold != 0)

        label_sums = Y[fold, :].sum(axis=0)

        if np.any(label_sums == 0):
            count += 1

    return count

def flz(Y, folds, validation=True):
    D = Y.sum(axis=0)

    # Remove only-negative labels
    Y = Y[:, D > 0]
    D = D[D > 0]

    count = 0
    for fold in folds:
        if validation:
            fold = np.logical_not(fold)
        else:
            fold = (fold != 0)

        label_sums = Y[fold, :].sum(axis=0)

        count += (label_sums == 0).sum()

    return count

def meanminp(W, folds, validation=False):
    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]

    D = 1.0/D

    W = W*D[..., None]

    all_p = []

    for fold in folds:
        if validation:
            fold = 1 - fold

        tmp = W @ fold
        p = tmp.min()
        all_p.append(p)

    return np.mean(all_p)


