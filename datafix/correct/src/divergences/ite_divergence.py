import time

import ite.cost as ic


def ite_div(reference, query, verbose=False, co=ic.MDJDist_DKL()):
    tic = time.time()
    skl = co.estimation(reference, query)

    toc = time.time()
    if verbose:
        print(
            "\ntime was ",
            toc - tic,
            " div is ",
            skl,
            " -- ",
            co,
            "\n",
        )

    return skl
