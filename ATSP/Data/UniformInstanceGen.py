import numpy as np
from GA import RunAllInstances

def UniformInstanceGen(Nj, Nm, low, high):
    T = np.random.randint(low=low, high=high, size=(Nj, Nm))
    M = np.expand_dims(np.arange(1, Nm+1), axis=0).repeat(repeats=Nj, axis=0)
    M = PermuteRows(M)
    JobAdj, MachineAdj, Makespan = RunAllInstances(T, M)

    return T, M, JobAdj, MachineAdj, Makespan


def PermuteRows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]

def Override(fn):
    """
    override decorator
    """
    return fn