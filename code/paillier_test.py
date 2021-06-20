"""
"""
import random, sys, time
from math import floor
from gmpy2 import mpz, powmod, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd, f_mod, f_div, sub, \
    mul, add

rand = random_state(random.randrange(sys.maxsize))


class PrivateKey(object):
    def __init__(self, p, q, n):
        if p == q:
            self.l = p * (p - 1)
        else:
            self.l = (p - 1) * (q - 1)
        try:
            self.m = invert(self.l, n)
        except ZeroDivisionError as e:
            print(e)
            exit()


class PublicKey(object):
    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits = mpz(rint_round(log2(self.n)))


def generate_prime(bits):
    """Will generate an integer of b bits that is prime using the gmpy2 library  """
    while True:
        possible = mpz(2) ** (bits - 1) + mpz_urandomb(rand, bits - 1)
        if is_prime(possible):
            return possible


def generate_keypair(bits):
    """ Will generate a pair of paillier keys bits>5"""
    p = generate_prime(bits // 2)
    # print(p)
    q = generate_prime(bits // 2)
    # print(q)
    n = p * q
    return PrivateKey(p, q, n), PublicKey(n)


def enc(pub, plain):  # (public key, plaintext)
    r = mpz_urandomb(random_state(random.randrange(sys.maxsize)), pub.bits)
    while (gcd(r, pub.n) != 1):
        r = mpz_urandomb(random_state(random.randrange(sys.maxsize)), pub.bits)

    return f_mod(mul(powmod(pub.g, plain, pub.n_sq), powmod(r, pub.n, pub.n_sq)), pub.n_sq)
    # return cipher


def dec(priv, pub, cipher):  # (private key, public key, cipher)
    x = powmod(cipher, priv.l, pub.n_sq)
    L = f_div(sub(x, 1), pub.n)
    return f_mod(mul(L, priv.m), pub.n)
    # return plain


def enc_add(pub, m1, m2):
    """Add one encrypted integer to another"""
    return f_mod(mul(enc(pub, m1), enc(pub, m2)), pub.n_sq)


def enc_add_const(pub, m, c):  # to do
    """Add constant n to an encrypted integer"""
    return f_mod(mul(powmod(pub.g, c, pub.n_sq), f_mod(enc(pub, m), pub.n_sq)), pub.n_sq)


def enc_mul_const(pub, m, c):  # to do
    """Multiplies an encrypted integer by a constant"""
    return powmod(enc(pub, m), c, pub.n_sq)


if __name__ == '__main__':
    priv, pub = generate_keypair(1024)
    """
    test
    """

    test_number = 100
    test_length = [10, 100, 500, 1000]
    tests = []
    tests_add = []
    all_tests_passed = True
    for j in range(len(test_length)):
        tests.append([])
        tests_add.append([])
        for i in range(test_number):
            tests[j].append(mpz_urandomb(random_state(random.randrange(sys.maxsize)), test_length[j]))
            tests_add[j].append(mpz_urandomb(random_state(random.randrange(sys.maxsize)), test_length[j]))
    # print(tests)
    test_enc_time = []
    test_dec_time = []
    for j in range(len(test_length)):
        test_dec_time.append(0)
        test_enc_time.append(0)
        for i in range(len(tests[j])):
            start = time.time()
            enc(pub, tests[j][i])
            end = time.time()
            test_enc_time[j] += end - start
            start = time.time()
            dec(priv, pub, tests[j][i])
            end = time.time()
            test_dec_time[j] += end - start
            if tests[j][i] + tests_add[j][i] != dec(priv, pub, enc_add(pub, tests[j][i], tests_add[j][i])):
                all_tests_passed = False
        test_enc_time[j] /= test_number
        test_dec_time[j] /= test_number
        print('Average Encrypt time in {} times for {} bits number: {}'.format(test_number, test_length[j],
                                                                               test_enc_time[j]))
        print('Average Decrypt time in {} times for {} bits number: {}'.format(test_number, test_length[j],
                                                                               test_dec_time[j]))
    if all_tests_passed:
        print("All add encrypt tests passed!")
