def init(loc):                  # returns Boundary
    return bnd

def solve(bnd1, bnd2):          # returns Patch
    return patch

def extract1(patch):            # returns Boundary
    return bnd
def extract2(patch):
    return bnd

def output(patch):              # returns None
    return None



def finit(loc):                 # returns Future[Boundary]
    return executor.submit(init, loc)

def fsolve(fbnd1, fbnd2):       # returns Future[Patch]
    return executor.submit(
        lambda: return solve(fbnd1.result(), fbnd2.result)))

def fextract1(fpatch):          # return Future[Boundary]
    return executor.submit(
        lambda: return extract1(fpatch.result()))
def fextract2(fpatch):          # return Future[Boundary]
    return executor.submit(
        lambda: return extract2(fpatch.result()))

def foutput(fpatch):            # returns None
    # We don't return a future since we want the output to occur
    # serialized, in a particular order, since all output presumably
    # goes into the same file. If we were to use a more sophisticated
    # output method, then this should also return a future, namely
    # Future[None].
    return output(fpatch.result())
