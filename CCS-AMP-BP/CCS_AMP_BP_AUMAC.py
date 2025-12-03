import numpy as np

def fht(u):
    """In-place fast Walshâ€“Hadamard transform for power-of-two length."""
    N = len(u)
    i = N >> 1
    while i:
        for j in range(N):
            if (i & j) == 0:
                temp = u[j]
                u[j] += u[i | j]
                u[i | j] = temp - u[i | j]
        i >>= 1


def sub_fht(n, m, seed=None, ordering=None, new_embedding=False):
    """
    Build a (subsampled) FHT pair Ax/Ay:
      Ax : R^m -> R^n
      Ay : R^n -> R^m
    """
    assert n > 0 and m > 0
    if new_embedding:
        w = 2 ** int(np.ceil(np.log2(max(m + 1, n + 1))))
    else:
        w = 2 ** int(np.ceil(np.log2(max(m, n + 1))))

    if ordering is None:
        rng = np.random.RandomState(seed)
        idxs = np.arange(1, w, dtype=np.uint32)
        rng.shuffle(idxs)
        ordering = idxs[:n]
    else:
        assert ordering.shape == (n,)

    def Ax(x):
        assert x.size == m
        y = np.zeros(w)
        if new_embedding:
            y[w - m :] = x.reshape(m)
        else:
            y[:m] = x.reshape(m)
        fht(y)
        return y[ordering]

    def Ay(y):
        assert y.size == n
        x = np.zeros(w)
        x[ordering] = y.reshape(n)
        fht(x)
        return x[w - m :] if new_embedding else x[:m]

    return Ax, Ay, ordering

from pyfhtWu import block_sub_fht_matrix as block_sub_fht  # type: ignore

def block_sub_fht(n, m, L, seed=None, ordering=None, new_embedding=False):
    """
    Block-diagonal Ax/Ay for L sections (maps R^{L*m} <-> R^n).
    """
    assert n > 0 and m > 0 and L > 0

    if ordering is None:
        if new_embedding:
            w = 2 ** int(np.ceil(np.log2(max(m + 1, n + 1))))
        else:
            w = 2 ** int(np.ceil(np.log2(max(m, n + 1))))
        rng = np.random.RandomState(seed)
        ordering = np.empty((L, n), dtype=np.uint32)
        idxs = np.arange(1, w, dtype=np.uint32)
        for ll in range(L):
            rng.shuffle(idxs)
            ordering[ll] = idxs[:n]
    else:
        assert ordering.shape == (L, n)

    def Ax(x):
        assert x.size == L * m
        out = np.zeros([n,1])
        for ll in range(L):
            ax, ay, _ = sub_fht(n, m, ordering=ordering[ll], new_embedding=new_embedding)
            out[:,0] += ax(x[ll * m : (ll + 1) * m])
        return out

    def Ay(y):
        assert y.size == n
        out = np.empty([L * m,1])
        for ll in range(L):
            ax, ay, _ = sub_fht(n, m, ordering=ordering[ll], new_embedding=new_embedding)
            out[ll * m : (ll + 1) * m,0] = ay(y)
        return out

    return Ax, Ay, ordering

def sparc_codebook(L, M, n, P):

    Ax, Ay, _ = block_sub_fht(n, M, L, seed=0, ordering=None)

    def Ab(b):
        B = np.asarray(b, dtype=np.float64)
        if B.ndim == 1:
            B = B.reshape(L * M, 1)
        else:
            assert B.shape[0] == L * M
        Y = Ax(B) / np.sqrt(n)           # (n, B)
        return Y

    def Az(z):
        Z = np.asarray(z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(n, 1)
        else:
            assert Z.shape[0] == n
        X = Ay(Z) / np.sqrt(n)           # (L*M, B)
        return X

    return Ab, Az


# ================================================================
#                        Outer (tree) code
# ================================================================
def Tree_encode(tx_message, Ka, messageBlocks, G, L, vl):
    """
    Encode Ka user messages (Ka x w bits) into L section indices using the tree code.
    messageBlocks marks info vs parity. G is the parity graph adjacency.
    Returns: (Ka x L) integer indices per section.
    """
    encoded = np.zeros((Ka, L), dtype=int)
    for i in range(L):
        if messageBlocks[i]:  # information section
            start = int(np.sum(messageBlocks[:i]) * vl)
            end = int((np.sum(messageBlocks[:i]) + 1) * vl)
            encoded[:, i] = tx_message[:, start:end].dot(2 ** np.arange(vl)[::-1])
        else:  # parity section
            idx = np.where(G[i])[0]
            parity = np.zeros((Ka, 1), dtype=int)
            for j in idx:
                parity = np.mod(parity + encoded[:, j].reshape(-1, 1), 2**vl)
            encoded[:, i] = parity.reshape(-1)
    return encoded


def convert_indices_to_sparse(encoded_tx_message_indices, L, vl, Ka):
    """
    (Ka x L) section indices -> (L*2^vl, 1) sparse counts (the usual SPARC input shape).
    """
    out = np.zeros((L * 2**vl, 1), dtype=int)
    for i in range(L):
        A = encoded_tx_message_indices[:, i]
        np.add.at(out, i * 2**vl + A.reshape(-1, 1), 1)
    return out


def convert_indices_to_sparse_asyn(encoded_tx_message_indices, L, vl, Ka):
    """
    (Ka x L) -> (L*2^vl, Ka) one-hot columns (per-user sparse vectors).
    """
    out = np.zeros((L * 2**vl, Ka), dtype=int)
    for k in range(Ka):
        for i in range(L):
            out[i * 2**vl + encoded_tx_message_indices[k, i], k] = 1
    return out


def convert_sparse_to_indices(score_vec, L, J, listSize):
    """
    Given a length-(L*2^J) score vector, take top listSize per section.
    Returns (listSize x L) indices.
    """
    cs = np.zeros((listSize, L), dtype=int)
    for i in range(L):
        A = score_vec[i * 2**J : (i + 1) * 2**J]
        idx = (A.reshape(2**J,)).argsort()[np.arange(2**J - listSize)]
        keep = np.setdiff1d(np.arange(2**J), idx)
        cs[:, i] = keep
    return cs


def extract_msg_indices(Paths, cs_decoded_tx_message, L, J):
    """
    Convert row-paths to actual section values from cs_decoded_tx_message.
    """
    msgs = np.empty((0, 0))
    R = Paths.shape[0]
    for i in range(R):
        path = Paths[i].reshape(1, -1)
        row = None
        for j in range(path.shape[1]):
            val = cs_decoded_tx_message[path[0, j], j]
            row = np.hstack((row, val.reshape(1, -1))) if row is not None else val.reshape(1, -1)
        msgs = np.vstack((msgs, row)) if msgs.size else row
    return msgs

def Tree_decoder(cs_decoded_tx_message,G,L,J,B,listSize):
    
    tree_decoded_tx_message = np.empty(shape=(0,0))
    
    Paths012 = merge_paths(cs_decoded_tx_message[:,0:3])
    
    Paths345 = merge_paths(cs_decoded_tx_message[:,3:6])
    
    Paths678 = merge_paths(cs_decoded_tx_message[:,6:9])
    
    Paths91011 = merge_paths(cs_decoded_tx_message[:,9:12])
    
    Paths01267812 = merge_pathslevel2(Paths012,Paths678,cs_decoded_tx_message[:,[0,6,12]])
    
    Paths3459101113 = merge_pathslevel2(Paths345,Paths91011,cs_decoded_tx_message[:,[3,9,13]])
    
    Paths01267812345910111314 = merge_all_paths0(Paths01267812,Paths3459101113,cs_decoded_tx_message[:,[1,4,10,14]])
    
    Paths = merge_all_paths_final(Paths01267812345910111314,cs_decoded_tx_message[:,[7,10,15]])
    
    
   
    return Paths

def merge_paths(A):
    listSize = A.shape[0]
    B = np.array([np.mod(A[:,0] + a,2**16) for a in A[:,1]]).flatten()
     
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,listSize).reshape(-1,1),np.floor(I/listSize).reshape(-1,1)]).astype(int)
            Paths = np.vstack((Paths,np.hstack([I1,np.repeat(i,I.shape[0]).reshape(-1,1)]))) if Paths.size else np.hstack([I1,np.repeat(i,I.shape[0]).reshape(-1,1)])
    
    return Paths

def merge_pathslevel2(Paths012,Paths678,A):
    listSize = A.shape[0]
    Paths0 = Paths012[:,0]
    Paths6 = Paths678[:,0]
    B = np.array([np.mod(A[Paths0,0] + a,2**16) for a in A[Paths6,1]]).flatten()
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,Paths0.shape[0]).reshape(-1,1),np.floor(I/Paths0.shape[0]).reshape(-1,1)]).astype(int)
            PPaths = np.hstack((Paths012[I1[:,0]].reshape(-1,3),Paths678[I1[:,1]].reshape(-1,3),np.repeat(i,I1.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
               
    return Paths

def merge_all_paths0(Paths01267812,Paths3459101113,A):
    listSize = A.shape[0]
    Paths1 = Paths01267812[:,1]
    Paths4 = Paths3459101113[:,1]
    Paths10 = Paths3459101113[:,4]
    Aa = np.mod(A[Paths4,1]+A[Paths10,2],2**16)
    B = np.array([np.mod(A[Paths1,0] + a,2**16) for a in Aa]).flatten()
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,3])[0].reshape(-1,1)
        if I.size:
            I1 = np.hstack([np.mod(I,Paths1.shape[0]).reshape(-1,1),np.floor(I/Paths1.shape[0]).reshape(-1,1)]).astype(int)
            PPaths = np.hstack((Paths01267812[I1[:,0]].reshape(-1,7),Paths3459101113[I1[:,1]].reshape(-1,7),np.repeat(i,I1.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
    
    return Paths

def merge_all_paths_final(Paths01267812345910111314,A):
    
    listSize = A.shape[0]
    Paths7 = Paths01267812345910111314[:,4]
    Paths10 = Paths01267812345910111314[:,11]
    B = np.mod(A[Paths7,0] + A[Paths10,1] ,2**16)
    
    Paths=np.empty((0,0))
    
    for i in range(listSize):
        I = np.where(B==A[i,2])[0].reshape(-1,1)
        if I.size:
            PPaths = np.hstack((Paths01267812345910111314[I].reshape(-1,15),np.repeat(i,I.shape[0]).reshape(-1,1)))
            Paths = np.vstack((Paths,PPaths)) if Paths.size else PPaths
    return Paths
# ================================================================
#              Outer-graph prior (BP as in your code)
# ================================================================
def approximateVector(x, K):
    """
    Project x onto an admissible set with at most K mass per section and L1=1.
    """
    x = np.maximum(x, 0)
    s = np.linalg.norm(x, 1)
    xOrig = x / (s if s > 0 else 1.0)

    xHt = xOrig.copy()
    while np.amax(xHt) > (1.0 / K):
        cap = (1.0 / K) * np.ones_like(xHt)
        minIndices = np.argmin([cap, xHt], axis=0)
        xHt = np.minimum(cap, xHt)

        deficit = 1.0 - np.linalg.norm(xHt, 1)
        if deficit <= 0:
            break
        mIx = np.linalg.norm((xHt * minIndices), 1)
        if mIx <= 0:
            break
        scale = (deficit + mIx) / mIx
        xHt = scale * (minIndices * xHt) + (1.0 / K) * (np.ones_like(xHt) - minIndices)
    return xHt


def computePrior(s, G, messageBlocks, L, M, p0, K, tau, Phat, numBPiter):
    """
    Belief-propagation-based prior 
    """
    # using global ml in message tensor sizes (consistent with your original)
    q = np.zeros(s.shape, dtype=float)
    p1 = p0 * np.ones(s.shape, dtype=float)
    idxParity = np.where(messageBlocks == 0)[0]
    mu_ap_sl = np.ones((L // 2 + 1, 4, ml))
    mu_sl_ap = np.ones((L, 4, ml))

    # Per-section PME from s
    temp_l = (p1 * np.exp(-(s - np.sqrt(Phat)) ** 2 / (2 * tau**2))) / (
        p1 * np.exp(-(s - np.sqrt(Phat)) ** 2 / (2 * tau**2)) + (1 - p1) * np.exp(-s**2 / (2 * tau**2))
    )
    lam = temp_l.reshape(L, -1)
    lam = lam / (np.sum(lam, axis=1).reshape(L, -1) + 1e-12)

    # BP iterations
    for _ in range(numBPiter):
        # variable -> factor
        for i in range(L):
            mu_sl_ap[i, :] = lam[i, :].copy()
            if not messageBlocks[i]:
                continue
            N_sl = np.where(G[i])[0]
            for a in N_sl:
                Nsl_minus_a = np.setdiff1d(N_sl, a)
                cc = np.where(N_sl == a)[0][0]
                for p in Nsl_minus_a:
                    r = np.where(idxParity == p)[0][0]
                    Nap = np.where(G[p])[0]
                    c = np.where(Nap == i)[0][0]
                    mu_sl_ap[i, cc] *= mu_ap_sl[r, c]
                mu_sl_ap[i, cc] = approximateVector(mu_sl_ap[i, cc], K)

        # factor -> variable
        for i in range(L):
            if messageBlocks[i]:
                N_si = np.where(G[i])[0]
                for ap in N_si:
                    N_ap = np.setdiff1d(np.where(G[ap])[0], i)
                    lamfft = np.fft.fft(mu_sl_ap[ap, 0])
                    for ak in N_ap:
                        r = ak
                        tmp = np.where(G[r])[0]
                        c = np.where(tmp == ap)[0][0]
                        flipped = np.hstack((mu_sl_ap[r, c][0].reshape(-1), np.flip(mu_sl_ap[r, c][1:])))
                        lamfft = np.vstack((lamfft, np.fft.fft(flipped)))
                    lamfft = np.prod(lamfft, axis=0)
                    r = np.where(idxParity == ap)[0][0]
                    tmp = np.where(G[ap])[0]
                    c = np.where(tmp == i)[0][0]
                    mu_ap_sl[r, c] = np.fft.ifft(lamfft).real
            else:
                N_si = np.where(G[i])[0]
                lamfft = None
                for sj in N_si:
                    r = sj
                    tmp = np.where(G[r])[0]
                    c = np.where(tmp == i)[0][0]
                    cur = np.fft.fft(mu_sl_ap[r, c])
                    lamfft = cur if lamfft is None else np.vstack((lamfft, cur))
                lamfft = np.prod(lamfft, axis=0)
                r = np.where(idxParity == i)[0][0]
                c = len(N_si)
                mu_ap_sl[r, c] = np.fft.ifft(lamfft).real

    # final priors per section
    for i in range(L):
        if messageBlocks[i]:
            N_si = np.where(G[i])[0]
            lamsec = None
            for ap in N_si:
                r = np.where(idxParity == ap)[0][0]
                tmp = np.where(G[ap])[0]
                c = np.where(tmp == i)[0][0]
                lamsec = mu_ap_sl[r, c] if lamsec is None else np.vstack((lamsec, mu_ap_sl[r, c]))
            lamsec = np.prod(lamsec, axis=0)
        else:
            N_si = np.where(G[i])[0]
            r = np.where(idxParity == i)[0][0]
            c = len(N_si)
            lamsec = mu_ap_sl[r, c]

        p1[i * M : (i + 1) * M] = (lamsec / (np.sum(lamsec) + 1e-12)).reshape(-1, 1)
        p1[i * M : (i + 1) * M] = 1 - (1 - p1[i * M : (i + 1) * M]) ** K

    q = np.minimum(p1, 1)
    return q



# ================================================================
#     Unique-delay async operator + single AMP over all groups
# ================================================================
def _slice_exact_len(y_async, start, N):
    """
    Return y_async[start:start+N] with zero padding
    """
    start = int(start)
    seg = np.asarray(y_async[start : start + N]).reshape(-1, 1)
    if seg.shape[0] < N:
        pad = np.zeros((N - seg.shape[0], 1), dtype=seg.dtype)
        seg = np.vstack([seg, pad])
    elif seg.shape[0] > N:
        seg = seg[:N]
    return seg




# ================================================================
#                       unknown delay
# =================================================================
def amp_async_unknown_delays(
    y_async, P, L, ml, numAmpIter,
    Ab_sync, Az_sync, N_sync, dm, delays, Ka,
    G, messageBlocks, numBPiter=2
):
    # Single AMP that decodes ALL users jointly. 
    
    eps=1e-12 # to avoid x/0. 

    Lml = L * ml    
    def Ab_ud(B, Lml,delay):
        y = np.zeros((int(N_sync + dm), 1))
        for g, dd in enumerate(delay):
            b = B[g * Lml : (g + 1) * Lml]
            y[int(dd) : int(dd) + N_sync, :] += Ab_sync(b)
        return y

    def Az_ud(y_async, Lml,delay):
        out = np.zeros((len(delay) * Lml, 1))
        for g, dd in enumerate(delay):
            seg = _slice_exact_len(y_async, int(dd), N_sync)
            out[g * Lml : (g + 1) * Lml, :] = Az_sync(seg)
        return out
    
    # AMP state
    beta = np.zeros(((1+dm) * Lml, 1))
    z = y_async.copy()
    Phat = N_sync * P / L                       # consistent with sync scaling
    N_async = int(N_sync + dm)
    p0_g = 1.0 - (1.0 - 1.0 / ((1+dm)*ml)) ** (Ka) # expected occupancy per section for that delay group
    # p0_g = 1.0 - (1.0 - 1.0 / ((1+dm)*ml)) ** (1.5*Ka)
    numAmpIter1=2
    numAmpIter2=numAmpIter-numAmpIter1
    delay_all=range(dm+1)
    for _ in range(numAmpIter1):
        tau = float(np.sqrt(np.sum(z**2) / max(N_async, 1)))
        s = (np.sqrt(Phat) * beta + Az_ud(z, Lml,delay_all)).astype(np.float64)

        # priors group-by-group and concatenate into q
        q = np.zeros_like(s)
        for g, K_g in enumerate(delay_all):
            s_g = s[g * Lml : (g + 1) * Lml]
            q_g = computePrior(s_g, G, messageBlocks, L, ml, p0_g, Ka/(1+dm), tau, Phat, numBPiter)
            q[g * Lml : (g + 1) * Lml] = q_g

        # denoiser
        num = q * np.exp(-(s - np.sqrt(Phat)) ** 2 / (2 * (tau**2 + eps)))
        den = num + (1 - q) * np.exp(-s**2 / (2 * (tau**2 + eps)))
        beta_new = (num / np.maximum(den, eps)).astype(float)

        # Residual 
        z = y_async - np.sqrt(Phat) * Ab_ud(beta_new, Lml, delay_all) + \
            (z / (N_async * (tau**2 + eps))) * (Phat * float(np.sum(beta_new)) - Phat * float(np.sum(beta_new**2)))

        beta = beta_new
    temp=beta.reshape(1+dm,Lml).sum(-1)
    delay_est = np.where(temp > 10)[0]
    if delay_est.shape[0] >= Ka:
        delay_est = np.argsort(temp)[-Ka:][::-1]
    delay_est=np.sort(delay_est)
    K_per=assign_kper(temp[delay_est],Ka)
    beta=beta.reshape(1+dm,Lml)[delay_est,:].reshape(-1,1)
    for _ in range(numAmpIter2):
        tau = float(np.sqrt(np.sum(z**2) / max(N_async, 1)))
        s = (np.sqrt(Phat) * beta + Az_ud(z, Lml,delay_est)).astype(np.float64)

        # priors group-by-group and concatenate into q
        q = np.zeros_like(s)
        for g, K_g in enumerate(K_per):
            if K_g !=0:
                s_g = s[g * Lml : (g + 1) * Lml]
                p0_g = 1.0 - (1.0 - 1.0 / ml) ** int(K_g)   
                q_g = computePrior(s_g, G, messageBlocks, L, ml, p0_g, int(K_g), tau, Phat, numBPiter)
                q[g * Lml : (g + 1) * Lml] = q_g

        # denoiser
        num = q * np.exp(-(s - np.sqrt(Phat)) ** 2 / (2 * (tau**2 + eps)))
        den = num + (1 - q) * np.exp(-s**2 / (2 * (tau**2 + eps)))
        beta_new = (num / np.maximum(den, eps)).astype(float)

        # Residual 
        z = y_async - np.sqrt(Phat) * Ab_ud(beta_new, Lml,delay_est) + \
            (z / (N_async * (tau**2 + eps))) * (Phat * float(np.sum(beta_new)) - Phat * float(np.sum(beta_new**2)))
        beta = beta_new
        temp=beta.reshape(delay_est.shape[0],Lml).sum(-1)
        K_per=assign_kper(temp,Ka)
    return beta, delay_est, K_per


def assign_kper(temp,Ka):
    K_per=np.zeros_like(temp)
    for z in np.flip(range(1, 10)):
        if K_per.sum()<Ka:
            K_per[temp>=L*(z-1)+L*2/3]+=1
        else:
            idx=np.where(temp>=L*(z-1)+L*2/3)
            idx_sort=np.argsort(temp[idx])[-len(idx):][::-1]
            for g in idx_sort:
                K_per[g]+=1
                if K_per.sum()>=Ka:
                    break
    return K_per# ================================================================
#                       Path scoring helper
# ================================================================
def pick_topKminusdelta_paths(Paths, cs_decoded_tx_message, beta, J, K, delta):

    R = Paths.shape[0]
    if R == 0:
        return Paths
    M = 1 << J
    scores = np.zeros(R, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64).reshape(-1)
    for i in range(R):
        s = 0.0
        for j in range(Paths.shape[1]):
            idx_row = Paths[i, j]
            val = cs_decoded_tx_message[idx_row, j]
            s += np.log(max(b[j * M + val], 1e-300))
        scores[i] = s
    T = max(0, min(R, K - delta))
    if T == 0:
        return np.empty((0, Paths.shape[1]), dtype=int)
    order = np.argsort(scores)
    keep = order[-T:]
    return Paths[keep, :]


# ================================================================
#                          Main experiment
# ================================================================
if __name__ == "__main__":
    # -------- system parameters --------
    Ka = 200                 # active users
    EbNodB_range = np.arange(9.1,9.4,0.8 )
    w = 128                 # payload bits per user
    L = 16                  # number of sections
    N = 38400               # channel uses per (sync) block
    vl = 16                 # bits per section index
    ml = 2 ** vl            # section size
    dm=384
    D = dm + 1
    # -------- simulation parameters --------
    numAmpIter = 4
    numBPiter = 2
    maxSims = 2
    perm = np.argsort(np.array([0,1,2,6,7,8,12,3,4,5,9,10,11,13,14,15]))
    
    # -------- outer tree code graph (same as your setup) --------
    messageBlocks = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]).astype(int)
    G = np.zeros((L, L)).astype(int)
    G[0, [2, 12]] = 1
    G[1, [2, 14]] = 1
    G[2, [0, 1]] = 1
    G[3, [5, 13]] = 1
    G[4, [5, 14]] = 1
    G[5, [3, 4]] = 1
    G[6, [8, 12]] = 1
    G[7, [8, 15]] = 1
    G[8, [6, 7]] = 1
    G[9, [11, 13]] = 1
    G[10, [11, 14, 15]] = 1
    G[11, [9, 10]] = 1
    G[12, [0, 6]] = 1
    G[13, [3, 9]] = 1
    G[14, [1, 4, 10]] = 1
    G[15, [7, 10]] = 1
    # -------- power / noise --------
    print(f'[maxSims :] {maxSims }')
    for EbNodB in EbNodB_range:
        EbNo = 10 ** (EbNodB / 10.0)
        P = 2 * w * EbNo / N
        sigma_n = 1.0
        Phat_sync = N * P / L
        Ab, Az = sparc_codebook(L, ml, N, P)

        # -------- stats --------
        detected_async = 0

        for sims in range(maxSims):
            # ----- generate/encode -----
            tx_message = np.random.randint(low=2, size=(Ka, w))
            enc_idx = Tree_encode(tx_message, Ka, messageBlocks, G, L, vl)

            # ----- asynchronous (single AMP over UNIQUE delays) -----
            delays = np.random.randint(low=0, high=dm + 1, size=Ka)

            # Build async observation by summing shifted user waveforms
            beta0_asyn_cols = convert_indices_to_sparse_asyn(enc_idx, L, vl, Ka)  # (L*ml, Ka)
            y_async = np.zeros((N + dm, 1))
            for k in range(Ka):
                dd = int(delays[k])
                y_async[dd : dd + N, :] += np.sqrt(Phat_sync) * Ab(beta0_asyn_cols[:, [k]])
            y_async += np.random.randn(N + dm, 1) * sigma_n

            beta_all,uniq_delays,K_per=amp_async_unknown_delays(
                y_async, P, L, ml, numAmpIter, 
                Ab_sync=Ab, Az_sync=Az, N_sync=N, dm=dm, delays=delays, Ka=Ka,
                G=G, messageBlocks=messageBlocks, numBPiter=numBPiter
            )
                
            decoded_all = np.empty((0, L), dtype=int)
            Lml = L * ml
            for g, dd in enumerate(uniq_delays):
                Kd = int(K_per[g])
                if Kd == 0:
                    continue
                listSize_d = max(1, 4 * Kd)
                beta_g = beta_all[g * Lml : (g + 1) * Lml]
                cs_g = convert_sparse_to_indices(beta_g, L, vl, listSize_d)
                try:
                    Paths_g = Tree_decoder(cs_g,G,L,vl,w,listSize_d)
                    Paths_g=Paths_g[:,perm]
                    if Paths_g.shape[0] > Kd:
                        Paths_g = pick_topKminusdelta_paths(Paths_g, cs_g, beta_g, vl, Kd, 0)
                    decoded_g = extract_msg_indices(Paths_g, cs_g, L, vl) if Paths_g.size else np.empty((0, L), int)
                    if decoded_g.size: # type: ignore
                        decoded_all = np.vstack((decoded_all, decoded_g)) # type: ignore
                except:
                    continue
            decoded_all_compare = decoded_all.copy()  
            for i in range(Ka):
                if decoded_all_compare.size == 0:
                    continue
                match = np.all(decoded_all_compare == enc_idx[i, :], axis=1)  
                if np.any(match):
                    detected_async += 1
                    decoded_all_compare = np.delete(decoded_all_compare, np.where(match)[0][0], axis=0)


        # ----- report -----
        per_user_err_async = (Ka * maxSims - detected_async) / (Ka * maxSims)
        print(f"Per-user probability of error (ASYNC) = {per_user_err_async:.6f}  (Eb/N0={EbNodB} dB, Ka={Ka})")

