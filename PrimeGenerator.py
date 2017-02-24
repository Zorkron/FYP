from sys import argv
import numpy as np

def prime(i, primes):
    for prime in primes:
        if not (i == prime or i % prime):
            return False
    primes.add(i)
    return i

def historic(n):
    primes = set([2])
    i, p = 2, 0
    while True:
        if prime(i, primes):
            p += 1
            if p == n:
                return primes
        i += 1

if __name__ == "__main__":
    primes = sorted(list(historic(int(argv[1]))))
    w, h = 25, 100
    Matrix1 = [[primes[x*100+y] for x in range(w)] for y in range(h)]
    np.savetxt("test.txt",np.matrix(Matrix1),fmt='%1d')


