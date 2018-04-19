def best_value(Y, sign=1):
    """
    Returns a vector whose components i are the minimum (default) or maximum of Y[:i]
    """

    Y_best = list(Y)
    global_val = Y[0]
    for i in range(len(Y_best)):
        if sign == 1:
            global_val = min(Y[i], global_val)
        else:
            global_val = max(Y[i], global_val)
        Y_best[i] = global_val
    return Y_best
