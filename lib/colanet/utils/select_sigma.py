
def select_sigma(sigma):
    sigmas = np.array([10, 15, 25, 30, 50, 70])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]
