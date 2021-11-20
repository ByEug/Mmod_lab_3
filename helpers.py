def get_free_probability(rho):
    return 1 - rho


def get_state_probs(rho, k):
    p0 = get_free_probability(rho)
    probs = [p0]
    for i in range(1, k + 1):
        probs.append((1 - rho) * (rho ** i))
    return probs


def get_theor_queue_len(rho):
    return (rho ** 2) / (1 - rho)

