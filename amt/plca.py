import numpy as np


def reconstruct(Pt, Pomega_p, Pp_t, omegan, tn):
    Pomega_t = np.zeros((omegan, tn))
    for i in range(tn):
        Pomega_t[:, i] = np.sum(Pomega_p * Pp_t[:, i], axis=1)
    return Pt * Pomega_t


def cross_entropy(U, V):
    return -np.sum(U * np.log(V))


def plca(Vo, pn, Pomega_p, maxstep=100, progress_step=50):
    eps = 1e-8
    step = 0

    V = np.abs(Vo)
    Pt = np.sum(V, axis=0)
    V = V / np.sum(V)
    omegan, tn = np.shape(V)

    Pp_t = np.random.rand(pn, tn)
    Pp_t /= np.sum(Pp_t, axis=0).reshape(1, -1)

    oldentropy = cross_entropy(V, reconstruct(Pt, Pomega_p, Pp_t, omegan, tn))

    Pp_omegat = np.zeros((pn, omegan, tn))

    while True:
        # E Step
        for p in range(pn):
            Pp_omegat[p, :, :] = Pomega_p[:, p].reshape(-1, 1) * Pp_t[p, :].reshape(1, -1)
        Pp_omegat /= np.sum(Pp_omegat, axis=0)

        # M Step
        for p in range(pn):
            Pp_t[p, :] = (np.sum(V * Pp_omegat[p, :, :], axis=0)) / (np.sum(V, axis=0))

        entropy = cross_entropy(V, reconstruct(Pt, Pomega_p, Pp_t, omegan, tn))

        impr = oldentropy - entropy
        if step > maxstep or (0 < impr < eps):
            break
        else:
            if step % progress_step == 0:
                print('Step %d: Entropy = %e, D(Entropy) = %e.\n' % (step, entropy, impr))
            oldentropy = entropy

        step += 1

    return Pt, Pp_t
