"""Sections 4-5 of the notebook: factors, the TAN Bayesian network and Variable Elimination."""

import itertools

import networkx as nx
import numpy as np


class Factor:
    def __init__(self, vars_, table):
        self.vars, self.table = list(vars_), np.asarray(table, dtype=float)

    def _expand(self, all_vars):
        t = self.table
        for v in all_vars:
            if v not in self.vars:
                t = t[..., None]
        order = self.vars + [v for v in all_vars if v not in self.vars]
        return np.transpose(t, [order.index(v) for v in all_vars])

    def multiply(self, other):
        all_vars = self.vars + [v for v in other.vars if v not in self.vars]
        return Factor(all_vars, self._expand(all_vars) * other._expand(all_vars))

    def marginalise(self, var):
        ax = self.vars.index(var)
        return Factor([v for v in self.vars if v != var], self.table.sum(axis=ax))

    def reduce(self, var, value):
        ax = self.vars.index(var)
        return Factor([v for v in self.vars if v != var], np.take(self.table, value, axis=ax))


class BayesNet:
    def __init__(self, parents, card, alpha=1.0):
        self.parents, self.card, self.alpha, self.cpt = parents, card, alpha, {}

    def fit(self, data):                              # CPT = smoothed relative frequencies
        for v, pa in self.parents.items():
            counts = np.full([self.card[p] for p in pa] + [self.card[v]], self.alpha)
            np.add.at(counts, tuple(data[c].to_numpy() for c in pa + [v]), 1)
            self.cpt[v] = Factor(pa + [v], counts / counts.sum(axis=-1, keepdims=True))
        return self

    def loglik(self, data):                           # log P(features | Anomaly = c), shape (n, 2)
        out = np.zeros((len(data), 2))
        for c in (0, 1):
            for v, pa in self.parents.items():
                if v == "anomaly":
                    continue
                idx = tuple(np.full(len(data), c) if p == "anomaly" else data[p].to_numpy() for p in pa)
                out[:, c] += np.log(self.cpt[v].table[idx + (data[v].to_numpy(),)])
        return out

    def predict_proba(self, data):                    # P(Anomaly = 1 | all features), Bayes rule
        z = self.loglik(data) + np.log(self.cpt["anomaly"].table)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return (p / p.sum(axis=1, keepdims=True))[:, 1]

    def posterior(self, evidence, query="anomaly"):   # exact Variable Elimination, any partial evidence
        factors = list(self.cpt.values())
        for var, val in evidence.items():             # 1. reduce by evidence
            factors = [f.reduce(var, val) if var in f.vars else f for f in factors]
        for h in [v for v in self.parents if v != query and v not in evidence]:
            related = [f for f in factors if h in f.vars]       # 2. eliminate each hidden variable
            factors = [f for f in factors if h not in f.vars]
            prod = related[0]
            for f in related[1:]:
                prod = prod.multiply(f)
            factors.append(prod.marginalise(h))
        result = factors[0]
        for f in factors[1:]:
            result = result.multiply(f)
        t = result.table.ravel()
        return t / t.sum()                            # 3. normalise

    def contributions(self, codes: dict):
        """Per-variable log-likelihood ratio log P(x_v | pa, A=1) - log P(x_v | pa, A=0).

        Positive values push the posterior towards Anomalous. These are evidence weights in the
        learned model, not causes.
        """
        out = {}
        for v, pa in self.parents.items():
            if v == "anomaly":
                continue
            terms = []
            for c in (0, 1):
                idx = tuple(c if p == "anomaly" else codes[p] for p in pa) + (codes[v],)
                terms.append(np.log(self.cpt[v].table[idx]))
            out[v] = float(terms[1] - terms[0])
        return out


def entropy(data, cols):
    p = data.groupby(cols).size().to_numpy() / len(data)
    return -(p * np.log(p)).sum()


def mutual_info(data, a, b, cond=None):               # I(a;b) or I(a;b | cond)
    if cond is None:
        return entropy(data, [a]) + entropy(data, [b]) - entropy(data, [a, b])
    return (entropy(data, [a, cond]) + entropy(data, [b, cond])
            - entropy(data, [a, b, cond]) - entropy(data, [cond]))


def learn_tan(data, features, cls="anomaly"):
    G = nx.Graph()
    for a, b in itertools.combinations(features, 2):
        G.add_edge(a, b, weight=mutual_info(data, a, b, cls))
    tree = nx.maximum_spanning_tree(G)
    root = max(features, key=lambda f: mutual_info(data, f, cls))
    parents = {cls: [], root: [cls]}
    for par, child in nx.bfs_edges(tree, root):          # same order as bfs_predecessors
        parents[child] = [cls, par]
    return parents, tree, root


def tree_layout(T, root):                             # simple top-down layout for tree-shaped graphs
    pos, nxt = {}, [0]

    def visit(n, parent, depth):
        kids = [c for c in T.neighbors(n) if c != parent]
        if not kids:
            pos[n] = (nxt[0], -depth)
            nxt[0] += 1
        else:
            for c in kids:
                visit(c, n, depth + 1)
            pos[n] = (np.mean([pos[c][0] for c in kids]), -depth)

    visit(root, None, 0)
    return pos


def tan_layout(tree, root):
    """Figure 1 layout: Anomaly on top, learned feature tree underneath."""
    pos = tree_layout(tree, root)
    pos = {n: (x, 1.5 * (y - 1)) for n, (x, y) in pos.items()}
    pos["anomaly"] = (np.mean([p[0] for p in pos.values()]), 1.0)
    return pos
