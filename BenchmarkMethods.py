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

import os
import sys
import re
import numpy as np
import pandas as pd
import argparse
from arffToDataFrame import convertToDataFrame
from bs4 import BeautifulSoup
import time
import json
import gzip
import gc
from contextlib import contextmanager

# Methods to benchmark
from RandomSplit import BalancedCrossValidation, BalancedCrossValidationGlobal, BalancedCrossValidationIncremental, BalancedCrossValidationIncrementalGlobal
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import PaperMetrics as pm
import multiprocessing as mp

class RunBalancedCrossValidation:
    def __init__(self, W, X, Y, nfolds, runs, tries, aggregator, store_folds=False):
        self.W = W
        self.X = X
        self.Y = Y
        self.nfolds = nfolds
        self.runs = runs
        self.tries = tries
        self.aggregator = aggregator
        self.store_folds = store_folds

    def __call__(self, i):
        seed = i+1

        folds, res = BalancedCrossValidation(self.W, self.nfolds, tries=self.tries, aggregator=self.aggregator, random_state=seed)
        #all_res.append(aggregator(res))

        res = self.aggregator(res)

        ld = pm.labels_distribution(self.Y, folds)
        ld10 = pm.labels_distribution(self.Y, folds, lam=10)
        ldtrain = pm.labels_distribution(self.Y, folds, validation=False)
        ed = pm.examples_distribution(self.Y, folds)
        fz = pm.fz(self.Y, folds)
        flz = pm.flz(self.Y, folds)
        meanminp = pm.meanminp(self.W, folds)

        if self.store_folds:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp,  "folds": folds}
        else:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp}

class RunBalancedCrossValidationIncremental:
    def __init__(self, W, X, Y, nfolds, runs, tries, aggregator, store_folds=False):
        self.W = W
        self.X = X
        self.Y = Y
        self.nfolds = nfolds
        self.runs = runs
        self.tries = tries
        self.aggregator = aggregator
        self.store_folds = store_folds

    def __call__(self, i):
        seed = i+1

        folds, res = BalancedCrossValidationIncremental(self.W, self.nfolds, tries=self.tries, random_state=seed)
        #all_res.append(aggregator(res))

        res = self.aggregator(res)

        ld = pm.labels_distribution(self.Y, folds)
        ld10 = pm.labels_distribution(self.Y, folds, lam=10)
        ldtrain = pm.labels_distribution(self.Y, folds, validation=False)
        ed = pm.examples_distribution(self.Y, folds)
        fz = pm.fz(self.Y, folds)
        flz = pm.flz(self.Y, folds)
        meanminp = pm.meanminp(self.W, folds)

        if self.store_folds:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp,  "folds": folds}
        else:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp}

class RunBalancedCrossValidationGlobal:
    def __init__(self, W, X, Y, nfolds, runs, aggregator, store_folds=False, device="cpu"):
        self.W = W
        self.X = X
        self.Y = Y
        self.nfolds = nfolds
        self.runs = runs
        self.aggregator = aggregator
        self.store_folds = store_folds
        self.device = device

    def __call__(self, i):
        #print("PID", os.getpid(), "start", time.time(), flush=True)
        seed = i+1

        folds, res = BalancedCrossValidationGlobal(self.W, self.nfolds, random_state=seed, device=self.device)
        #all_res.append(aggregator(res))

        res = self.aggregator(res)

        ld = pm.labels_distribution(self.Y, folds)
        ld10 = pm.labels_distribution(self.Y, folds, lam=10)
        ldtrain = pm.labels_distribution(self.Y, folds, validation=False)
        ed = pm.examples_distribution(self.Y, folds)
        fz = pm.fz(self.Y, folds)
        flz = pm.flz(self.Y, folds)
        meanminp = pm.meanminp(self.W, folds)

        if self.store_folds:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp,  "folds": folds}
        else:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp}

class RunBalancedCrossValidationIncrementalGlobal:
    def __init__(self, W, X, Y, nfolds, runs, aggregator, store_folds=False, device="cpu"):
        self.W = W
        self.X = X
        self.Y = Y
        self.nfolds = nfolds
        self.runs = runs
        self.aggregator = aggregator
        self.store_folds = store_folds
        self.device = device

    def __call__(self, i):
        #print("PID", os.getpid(), "start", time.time(), flush=True)
        seed = i+1

        folds, res = BalancedCrossValidationIncrementalGlobal(self.W, self.nfolds, random_state=seed, device=self.device)
        #all_res.append(aggregator(res))

        res = self.aggregator(res)

        ld = pm.labels_distribution(self.Y, folds)
        ld10 = pm.labels_distribution(self.Y, folds, lam=10)
        ldtrain = pm.labels_distribution(self.Y, folds, validation=False)
        ed = pm.examples_distribution(self.Y, folds)
        fz = pm.fz(self.Y, folds)
        flz = pm.flz(self.Y, folds)
        meanminp = pm.meanminp(self.W, folds)

        if self.store_folds:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp,  "folds": folds}
        else:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp}

class RunMultilabelStratifiedKFold:
    def __init__(self, W, Worig, X, Y, nfolds, runs, tries, aggregator, store_folds=False):
        self.W = W
        self.Worig = Worig
        self.X = X
        self.Y = Y
        self.nfolds = nfolds
        self.runs = runs
        self.tries = tries
        self.aggregator = aggregator
        self.store_folds = store_folds

    def __call__(self, i):
        seed = i+1

        random_state = np.random.RandomState(seed) # XXX: Prevent MultilabelStratifiedKFold from starting over
        mskf = MultilabelStratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=random_state)
        
        bestRes = None
        bestFolds = None

        #for j in range(self.tries):
        for j in range(1):
            res = []
            folds = []

            for _, ind in mskf.split(self.Y, self.Y):
                x = np.ones(self.Y.shape[0], dtype=int)
                x[ind] = 0
                
                folds.append(x)
                res.append(np.linalg.norm(np.inner(self.W, x)))

            if bestRes is None or self.aggregator(res) < self.aggregator(bestRes):
                bestRes = res
                bestFolds = folds

        #all_res.append(aggregator(bestRes))
        bestRes = self.aggregator(bestRes)

        ld = pm.labels_distribution(self.Y, bestFolds)
        ld10 = pm.labels_distribution(self.Y, bestFolds, lam=10)
        ldtrain = pm.labels_distribution(self.Y, folds, validation=False)
        ed = pm.examples_distribution(self.Y, bestFolds)
        fz = pm.fz(self.Y, folds)
        flz = pm.flz(self.Y, folds)
        meanminp = pm.meanminp(self.Worig, folds)
        #print(f"LD: {ld}, ED: {ed}, FZ: {fz}, FLZ: {flz}")

        if self.store_folds:
            return {"res": bestRes, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp, "folds": folds}
        else:
            return {"res": bestRes, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp}

class RunPureRandomCrossValidation:
    def __init__(self, W, X, Y, nfolds, runs, tries, aggregator, store_folds=False):
        self.W = W
        self.X = X
        self.Y = Y
        self.nfolds = nfolds
        self.runs = runs
        self.tries = tries
        self.aggregator = aggregator
        self.store_folds = store_folds

    def __call__(self, i):
        seed = i+1

        folds, res = PureRandomCrossValidation(self.W, self.nfolds, tries=self.tries, aggregator=self.aggregator, random_state=seed)

        res = self.aggregator(res)

        ld = pm.labels_distribution(self.Y, folds)
        ld10 = pm.labels_distribution(self.Y, folds, lam=10)
        ldtrain = pm.labels_distribution(self.Y, folds, validation=False)
        ed = pm.examples_distribution(self.Y, folds)
        fz = pm.fz(self.Y, folds)
        flz = pm.flz(self.Y, folds)
        meanminp = pm.meanminp(self.W, folds)

        if self.store_folds:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp, "folds": folds}
        else:
            return {"res": res, "LD": ld, "LD10": ld10, "LDTrain": ldtrain, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp}

@contextmanager
def catchtime(prefix="Elapsed time:"):
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    print(f"{prefix} {time.perf_counter() - start:0.3f} seconds")

def _load_blca_data(path, use_rep_slide=False):
    data = pd.read_csv(path, keep_default_na=False)

    rows = []

    for r in range(data.shape[0]):
        row = dict()

        if str(data["rep_slide"][r]).strip() == "exclude":
            continue

        if use_rep_slide and data["rep_slide"][r] == 0:
            continue

        for label in data.columns:
            if label not in { "patient_id", "pathTstage", "center_id" }:
                continue

            patient_id = str(data["patient_id"][r])

            if label == "patient_id":
                row[label] = patient_id
            elif label == "pathTstage":
                code = str(data[label][r]).lower().strip()
                responder = int(data["label_binary"][r])

                # Filter out comments
                tmp = re.search("t(1?is|[0-4])[ab]?(n(x|[0-3]))?", code)

                if tmp:
                    code = tmp.group(0)
                else:
                    code = ""

                stage = 0
                lymph_node_count = 0

                if "t0" in code:
                    stage = 2
                elif "is" in code: # Covers tis and t1is
                    stage = 3
                elif "ta" in code:
                    stage = 4
                elif "t1" in code:
                    stage = 5
                elif "t2" in code:
                    stage = 6
                #elif "t3a" in code:
                #    stage = 7
                #elif "t3b" in code:
                #    stage = 8
                elif "t3" in code: # Stephanie says to ignore a/b for T3
                    stage = 7
                elif "t4" in code:
                    stage = 8 # Was 9
                elif len(code) > 0:
                    raise RuntimeError(f"Present pathTstage field but unable to interpret code for patient {patient_id}: '{code}'")

                if "n0" in code:
                    lymph_node_count = 0
                elif "n1" in code:
                    lymph_node_count = 1
                elif "n2" in code:
                    lymph_node_count = 2
                elif "n3" in code:
                    lymph_node_count = 3
                elif "nx" in code:
                    lymph_node_count = 4

                if stage == 0 and responder == 0:
                    stage = 1 # Cancer, unknown stage

                stage_key = f"stage{stage}"
                lymph_node_count_key = f"lymph_node{lymph_node_count}"

                row[stage_key] = 1
                row[lymph_node_count_key] = 1

                for i in range(9):
                    row.setdefault(f"stage{i}", 0)

                for i in range(5):
                    row.setdefault(f"lymph_node{i}", 0)

                row["label_binary"] = responder

            elif label == "center_id":
                center_id = int(data[label][r])

                center_id_key = f"center{center_id}"

                row[center_id_key] = 1

                for i in range(7):
                    row.setdefault(f"center{i}", 0)

            else:
                raise RuntimeError(f"Unrecognized BLCA column '{label}'")

            #End of for loop

        rows.append(row)

    new_data = dict()

    for row in rows:
        patient_id = row["patient_id"]

        row["nslides"] = 1

        if patient_id not in new_data:
            new_data[patient_id] = dict(row)
        else:
            for key in row:
                if key != "patient_id":
                    new_data[patient_id][key] += row[key]

    new_data = [ row for row in new_data.values() ]

    new_data = pd.DataFrame(new_data)

    labels = sorted([ label for label in new_data.columns if label != "patient_id" ])

    return new_data, labels

def load_data(path):
    if "blca" in os.path.basename(path):
        return _load_blca_data(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".arff":
        # Some mulan data has train/test, load both in that case
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        if re.search("-(train|test)", basename):
            print("Info: Loading train/test .arff files.")
            
            train_basename = re.sub("-test", "-train", basename)
            test_basename = re.sub("-train", "-test", basename)
            xml_basename = os.path.splitext(re.sub("-(train|test)", "", basename))[0] + ".xml"
            
            train_path = os.path.join(dirname, train_basename)
            test_path = os.path.join(dirname, test_basename)
            xml_path = os.path.join(dirname, xml_basename)
            
            train = convertToDataFrame(train_path, contains_attributes=True)
            test = convertToDataFrame(test_path, contains_attributes=True)
            
            with open(xml_path, mode="r", newline="") as f:
                xml = BeautifulSoup(f, "xml")
                    
            labels = [ x.get("name").replace("'", r"\'") for x in xml.find_all("label") ]
        
            data = pd.concat((train, test), axis=0)
        
            assert all(label in data.columns for label in labels)
        
            return data, labels
        else:
            xml_path = os.path.splitext(path)[0] + ".xml"
            
            data = convertToDataFrame(path, contains_attributes=True)
            
            with open(xml_path, mode="r", newline="") as f:
                xml = BeautifulSoup(f, "xml")
                
            #labels = [ x.get("name").replace("'", r"\'") for x in xml.find_all("label") ]

            labels = [ x.get("name") for x in xml.find_all("label") ]

            assert all(label in data.columns for label in labels)
            
            return data, labels
    elif ext == ".csv":
        data = pd.read_csv(path)
        exclude_fields = { "Patient ID", "patient id", "Subject_ID", "age" }
        labels = [ label for label in data.columns if label not in exclude_fields ]
        
        return data, labels
    else:
        raise RuntimeError(f"Unsupported file extension '{ext}'. Only '.csv' and '.arff' are supported.")

def make_weight_matrix(data, labels, self_count=True, negative_labels=False):
    rows = []
    
    if self_count:
        rows.append(np.ones(data.shape[0])[None, :]) # Self-count
    
    for label in labels:
        row = np.asarray(data[label].astype(float))[None, :]
        rows.append(row)

        if negative_labels and row.max() > 0:
            assert row.min() >= 0 and row.max() <= 1
            row = 1 - row
            rows.append(row)

    return np.concatenate(tuple(rows), axis=0)
    
def make_xy(data, labels):
    labels = set(labels)
    
    X = []
    Y = []
    
    for label in data.columns:
        try:
            col = np.asarray(data[label].astype(float))[:, None]
        except ValueError:
            continue
        
        if label in labels:
            Y.append(col)
        else:
            X.append(col)

    if len(X) == 0:
        return np.zeros((data.shape[0], 1)), np.concatenate(tuple(Y), axis=1)   
    else:
        return np.concatenate(tuple(X), axis=1), np.concatenate(tuple(Y), axis=1)   

def groups_y(Y, bins="fd"):
    pass

def discretize_y(Y, bins="auto", self_count=False):
    N = Y.shape[0]
    K = Y.shape[1]
    new_K = 0
    
    for c in range(K):
        col = Y[:, c]
        
        min_value = np.min(col)
        max_value = np.max(col)
        
        int_res = np.max(np.abs(col - col.astype(int)))
        if int_res == 0:
            print(f"{c}: Discrete. ", end="")
            
            if min_value == max_value:
                print("One value")
                continue
            
            if min_value == 0 and max_value == 1:
                print("Binary")
                new_K += 1
                continue # Nothing to do
            
            one_hot_dim = int(max_value - min_value + 1)
            new_K += one_hot_dim
            print(f"One hot dimension: {one_hot_dim}")
        else:
            print(f"{c}: Not discrete. ", end="")
            
            if min_value == max_value:
                print("One value")
                continue

            bin_edges = np.histogram_bin_edges(col, bins=bins)
            
            print(f"Hist dimension: {bin_edges.size-1}")
            new_K += bin_edges.size-1
            
    new_Y = np.zeros_like(Y, shape=(Y.shape[0], new_K))
    print(new_Y.shape)
    
    k = 0
    for c in range(K):
        col = Y[:, c]
        
        min_value = np.min(col)
        max_value = np.max(col)
        
        int_res = np.max(np.abs(col - col.astype(int)))
        if int_res == 0:
            if max_value == min_value:
                continue
                
            if min_value == 0 and max_value == 1:
                new_Y[:, k] = col
                k += 1
                continue
                
            one_hot_dim = int(max_value - min_value + 1)
            
            ind = (k + (col - min_value)).astype(int)
            new_Y[np.arange(N), ind] = 1
            
            k += one_hot_dim
        else:
            if min_value == max_value:
                continue
                
            bin_edges = np.histogram_bin_edges(col, bins=bins)
            
            ind = (k + np.minimum(np.searchsorted(bin_edges, col, side="right")-1, bin_edges.size-2)).astype(int)
            
            new_Y[np.arange(N), ind] = 1
            k += bin_edges.size-1
            
    assert new_Y.min() >= 0 and new_Y.max() <= 1

    if self_count:
        new_Y = np.c_[new_Y, np.ones(Y.shape[0])]
    
    return new_Y

def benchmark_BalancedCrossValidation(W, X, Y, nfolds, runs, tries, aggregator=np.max, num_threads=1, store_folds=False):
    chunksize=1

    print(num_threads)

    with mp.Pool(num_threads) as p:
        all_metrics = list(p.imap_unordered(RunBalancedCrossValidation(W,X,Y,nfolds,runs,tries,aggregator, store_folds=store_folds), range(runs), chunksize))

    return all_metrics

def benchmark_BalancedCrossValidationIncremental(W, X, Y, nfolds, runs, tries,  aggregator=np.max, num_threads=1, store_folds=False):
    chunksize=1

    print(num_threads)

    with mp.Pool(num_threads) as p:
        all_metrics = list(p.imap_unordered(RunBalancedCrossValidationIncremental(W,X,Y,nfolds,runs,tries, aggregator, store_folds=store_folds), range(runs), chunksize))

    return all_metrics

def benchmark_BalancedCrossValidationGlobal(W, X, Y, nfolds, runs, aggregator=np.max, num_threads=1, store_folds=False, device="cpu"):
    chunksize=1

    if device != "cpu":
        print("Info: Given device is not a CPU, limiting number of threads to 1")
        num_threads = 1


    with mp.Pool(num_threads) as p:
        all_metrics = list(p.imap_unordered(RunBalancedCrossValidationGlobal(W,X,Y,nfolds,runs,aggregator, store_folds=store_folds, device=device), range(runs), chunksize))

    return all_metrics

def benchmark_BalancedCrossValidationIncrementalGlobal(W, X, Y, nfolds, runs, aggregator=np.max, num_threads=1, store_folds=False, device="cpu"):
    chunksize=1

    if device != "cpu":
        print("Info: Given device is not a CPU, limiting number of threads to 1")
        num_threads = 1

    with mp.Pool(num_threads) as p:
        all_metrics = list(p.imap_unordered(RunBalancedCrossValidationIncrementalGlobal(W,X,Y,nfolds,runs,aggregator, store_folds=store_folds, device=device), range(runs), chunksize))

    return all_metrics
   
def benchmark_MultilabelStratifiedKFold(W, X, Y, nfolds, runs, tries, aggregator=np.max, num_threads=None, store_folds=False):
    chunksize=1
    
    Worig = W.copy()

    D = W.sum(axis=1)
    W = W[D > 0, :]
    D = D[D > 0]
    D = 1.0/D
    
    W = W*D[..., None]
    
    Z = np.eye(W.shape[0]) - 1.0/W.shape[0]

    W = Z @ W

    with mp.Pool(num_threads) as p:
        all_metrics = list(p.imap_unordered(RunMultilabelStratifiedKFold(W,Worig,X,Y,nfolds,runs,tries,aggregator, store_folds=store_folds), range(runs), chunksize))
       
    return all_metrics
    
def PureRandomCrossValidation(W, F, tries=10, aggregator=np.max, random_state=None, max_batch_size=1000):
    assert W.ndim == 2
    
    N = W.shape[1]

    assert F > 1 and F <= N

    assert np.all(W.max(axis=0) > 0) # Make sure all instances count for something
    
    if random_state is None:
        rng = np.random.default_rng()
    else:
        try: 
            rng = np.random.default_rng(int(random_state))
        except TypeError:
            rng = random_state

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

    #ind = np.arange(N)
    #
    #for _ in range(tries):        
    #    #np.random.shuffle(ind)
    #    rng.shuffle(ind)

    #    res = []
    #    folds = []

    #    for f in range(F):
    #        val_begin = N*f//F
    #        val_end = N*(f+1)//F
    #    
    #        x = np.ones(N, dtype=int)
    #        x[ind[val_begin:val_end]] = 0
    #   
    #        folds.append(x)
    #        res.append(np.linalg.norm(np.inner(W, x)))
    #    
    #    if bestRes is None or aggregator(res) < aggregator(bestRes):
    #        bestRes = res
    #        bestFolds = folds
            
    return bestFolds, bestRes
    
def benchmark_PureRandomCrossValidation(W, X, Y, nfolds, runs, tries, aggregator=np.max, num_threads=None, store_folds=False):
    chunksize=1


    with mp.Pool(num_threads) as p:
        all_metrics = list(p.imap_unordered(RunPureRandomCrossValidation(W,X,Y,nfolds,runs,tries,aggregator, store_folds=store_folds), range(runs), chunksize))

    #all_metrics = []
    #for i in range(runs):
    #    seed = i+1
    #    np.random.seed(seed)
    #    folds, res = PureRandomCrossValidation(W, nfolds, tries=tries, aggregator=aggregator, random_state=seed)

    #    res = aggregator(res)

    #    ld = pm.labels_distribution(Y, folds)
    #    ed = pm.examples_distribution(Y, folds)
    #    fz = pm.fz(Y, folds)
    #    flz = pm.flz(Y, folds)
    #    meanminp = pm.meanminp(W, folds)
    #    all_metrics.append({"res": res, "LD": ld, "ED": ed, "FZ": fz, "FLZ": flz, "MeanMinP": meanminp, "folds": folds})
   
    return all_metrics

def compute_fold_variability(metrics):
    try:
        folds = [ metric["folds"] for metric in metrics ]
        return pm.variability_over_all_pairs(folds)
    except KeyError:
        return None

def print_report(metrics, varH = None):
    keys = [ "res", "LD", "LD10", "LDTrain", "ED", "FZ", "FLZ", "MeanMinP" ]


    for key in keys:
        values = np.asarray([ metric[key] for metric in metrics ])
        print(f"{key}: {values.mean()} +/- {values.std()}", flush=True)

    if varH is not None:
        print(f"varH: {varH}", flush=True)



def clean_for_json(arg):
    if isinstance(arg, dict):
        return { clean_for_json(key): clean_for_json(value) for key, value in arg.items() }
    elif isinstance(arg, tuple):
        return tuple( clean_for_json(value) for value in arg )
    elif isinstance(arg, list):
        return [ clean_for_json(value) for value in arg ]
    elif isinstance(arg, set):
        return { clean_for_json(value) for value in arg }
    elif isinstance(arg, np.integer):
        return int(arg)
    elif isinstance(arg, np.floating):
        return float(arg)
    elif isinstance(arg, np.ndarray):
        return [ clean_for_json(value) for value in list(arg) ]

    return arg

def main(data_path, nfolds, runs, tries, output=None, num_threads=None, store_folds=False, negative_labels=False, device="cpu"):
    assert tries > 0
    assert runs > 0
    assert nfolds > 1

    #if device != "cpu":
    #    print("Info: Given device is not a CPU, limiting number of threads to 1")
    #    num_threads = 1

    #os.system(f"taskset -p 0xffffffff {os.getpid()}")
    mp.set_start_method('spawn' if sys.platform == 'win32' else 'forkserver', force=True)

    if num_threads is not None and num_threads > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    aggregator = np.mean
    
    print(f"Info: Loading '{data_path}' ...")
    
    data, labels = load_data(data_path)
    
    print(f"Info: There are {len(labels)} labels to split.")
    print(f"Info: Data shape {data.shape}.")
    
    W = make_weight_matrix(data, labels, negative_labels=negative_labels)
    X, Y = make_xy(data, labels)
    
    Y = discretize_y(Y)

    stats = {}

    stats["Setup"] = { "data_path": data_path, "runs": runs, "tries": tries, "nfolds": nfolds }

    with catchtime("NullSplitKFold elapsed:") as t:
        all_metrics = benchmark_BalancedCrossValidation(W, X, Y, nfolds, runs, tries, aggregator=aggregator, num_threads=num_threads, store_folds=store_folds)
        run_time = t()

    varH = compute_fold_variability(all_metrics)

    print_report(all_metrics, varH = varH)

    stats["NullSplitKFold"] = {"metrics": all_metrics, "run_time": run_time}

    if varH is not None:
        stats["NullSplitKFold"]["varH"] = varH

    with catchtime("NullSplitKFoldIncremental elapsed:") as t:
        all_metrics = benchmark_BalancedCrossValidationIncremental(W, X, Y, nfolds, runs, tries, aggregator=aggregator, num_threads=num_threads, store_folds=store_folds)
        run_time = t()

    varH = compute_fold_variability(all_metrics)

    print_report(all_metrics, varH = varH)

    # XXX: This function used to be called this
    #stats["NullSplitKFoldIncrementalRandom"] = {"metrics": all_metrics, "run_time": run_time}
    stats["NullSplitKFoldIncremental"] = {"metrics": all_metrics, "run_time": run_time}

    if varH is not None:
        stats["NullSplitKFoldIncrementalRandom"]["varH"] = varH

    with catchtime("NullSplitKFoldGlobal elapsed:") as t:
        all_metrics = benchmark_BalancedCrossValidationGlobal(W, X, Y, nfolds, runs, aggregator=aggregator, num_threads=num_threads, store_folds=store_folds, device=device)
        run_time = t()

    if store_folds:
        all_metrics["varH"] = compute_fold_variability(all_metrics)

    print_report(all_metrics)

    stats["NullSplitKFoldGlobal"] = {"metrics": all_metrics, "run_time": run_time}

    with catchtime("NullSplitKFoldIncrementalGlobal elapsed:") as t:
        all_metrics = benchmark_BalancedCrossValidationIncrementalGlobal(W, X, Y, nfolds, runs, aggregator=aggregator, num_threads=num_threads, store_folds=store_folds, device=device)
        run_time = t()

    if store_folds:
        all_metrics["varH"] = compute_fold_variability(all_metrics)

    print_report(all_metrics)

    # XXX: This function used to be called this
    #stats["NullSplitKFoldIncremental"] = {"metrics": all_metrics, "run_time": run_time}
    stats["NullSplitKFoldIncrementalGlobal"] = {"metrics": all_metrics, "run_time": run_time}
   
    with catchtime("MultilabelStratifiedKFold elapsed:") as t:
        all_metrics = benchmark_MultilabelStratifiedKFold(W, X, Y, nfolds, runs, tries, aggregator=aggregator, num_threads=num_threads, store_folds=store_folds)
        run_time = t()

    varH = compute_fold_variability(all_metrics)

    print_report(all_metrics, varH = varH)

    stats["MultilabelStratifiedKFold"] = {"metrics": all_metrics, "run_time": run_time}

    if varH is not None:
        stats["MultilabelStratifiedKFold"]["varH"] = varH
       
    with catchtime("PureRandomKFold elapsed:") as t:
        all_metrics = benchmark_PureRandomCrossValidation(W, X, Y, nfolds, runs, tries, aggregator=aggregator, num_threads=num_threads, store_folds=store_folds)
        run_time = t()

    varH = compute_fold_variability(all_metrics)

    print_report(all_metrics, varH = varH)

    stats["PureRandomKFold"] = {"metrics": all_metrics, "run_time": run_time}

    if varH is not None:
        stats["PureRandomKFold"]["varH"] = varH

    stats = clean_for_json(stats)

    if output is not None:
        if os.path.splitext(output)[1].lower() == '.gz':
            with gzip.open(output, mode="wt", newline="") as f:
                json.dump(stats, f, indent=4)
        else:
            with open(output, mode="wt", newline="") as f:
                json.dump(stats, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NullSplit Benchmark Tool")
    parser.add_argument("--data-path", dest="data_path", required=True, type=str, help="Path to CSV or ARFF file.")
    parser.add_argument("--nfolds", dest="nfolds", required=True, type=int, help="Number of folds to split.")
    parser.add_argument("--runs", dest="runs", required=True, type=int, help="Number of split experiments to perform.")
    parser.add_argument("--tries", dest="tries", required=False, type=int, default=1, help="Split 'tries' times and keep the best split.")
    parser.add_argument("--output", dest="output", required=False, type=str, default=None, help="Output JSON path to store statistics.")
    parser.add_argument("--nthreads", dest="num_threads", required=False, type=int, default=None, help="Number of threads to use for benchmarking.")
    parser.add_argument("--store-folds", dest="store_folds", required=False, default=False, action="store_true", help="Store the folds in the JSON file.")
    parser.add_argument("--negative-labels", dest="negative_labels", required=False, default=False, action="store_true", help="Augment weight matrix to have negative label counts for NullSplit.")
    parser.add_argument("--device", dest="device", required=False, default="cpu", help="Try to use given device to speedup computation.")
    
    args = parser.parse_args()
    
    main(**vars(args))
    
