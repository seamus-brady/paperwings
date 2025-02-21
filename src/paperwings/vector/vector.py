# MIT License
#
# Based on original code from https://github.com/mansourkheffache/hdc by Mansour Kheffache.
# Modifications have been made by Seamus Brady in 2025.
#
# Copyright (c) 2018 Mansour Kheffache
# Copyright (c) 2025 Seamus Brady
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from datetime import datetime

import numpy as np

random.seed()  # nosec


class AbstractVector:
    """
    A template class for Vectors.
    Defines INIT, ADD, MUL, and DIST operations depending on
    representation in subclasses.
    """

    DEFAULT_VECTOR_SIZE = 1000
    BINARY_VECTOR_TYPE = "binary"
    BIPOLAR_VECTOR_TYPE = "bipolar"
    BINARY_SPARSE_VECTOR_TYPE = "binary_sparse"

    @staticmethod
    def new_vector(size=DEFAULT_VECTOR_SIZE, rep=BINARY_VECTOR_TYPE):
        if rep == AbstractVector.BINARY_VECTOR_TYPE:
            return BinaryVector(size)
        if rep == AbstractVector.BIPOLAR_VECTOR_TYPE:
            return BipolarVector(size)
        if rep == AbstractVector.BINARY_SPARSE_VECTOR_TYPE:
            return BinarySparseVector(size)

    def __init__(self, size):
        """
        Create a new (random) hyperdimensional vector
        """
        self.size = size
        self.value = None
        self.rep = None
        self.strength = 100
        self.created_at = datetime.now()

    def add(self, x, y):
        """
        Create a new hyperdimensional vector, z, that is the result of the
        addition (bundling) of two hyperdimensional vectors x and y.
        """
        return None

    def sub(self, x, y):
        """
        Create a new hyperdimensional vector, z, that is the result of the
        subtraction of two hyperdimensional vectors x and y.
        """
        return None

    def mul(self, x, y):
        """
        Create a new hyperdimensional vector, z, that is the result of the
        multiplication (binding) of two hyperdimensional vectors x and y.
        """
        return None

    def __dist__(self, x, y):
        """
        Return the distance between the two hyperdimensional vectors x and y.
        """
        return None

    # print vector
    def __repr__(self):
        return np.array2string(self.value)

    # print vector
    def __str__(self):
        return np.array2string(self.value)

    # addition
    def __add__(self, a):
        b = AbstractVector.new_vector(self.size, self.rep)
        b.value = self.add(self.value, a.value)
        return b

    # addition
    def __sub__(self, a):
        b = AbstractVector.new_vector(self.size, self.rep)
        b.value = self.sub(self.value, a.value)
        return b

    # multiplication
    def __mul__(self, a):
        b = AbstractVector.new_vector(self.size, self.rep)
        b.value = self.mul(self.value, a.value)
        return b

    # distance
    def dist(self, a):
        return self.__dist__(self.value, a.value)


class BinaryVector(AbstractVector):
    """
    A class representing a binary Vector.
    """

    def __init__(self, size):
        super().__init__(size)
        self.rep = AbstractVector.BINARY_VECTOR_TYPE
        # noinspection PyTypeChecker
        self.value = np.random.randint(2, size=size)

    def add(self, x, y):
        z = x + y
        z[z == 1] = np.random.randint(2, size=len(z[z == 1]))
        z[z == 2] = np.ones(len(z[z == 2]))
        return z

    def sub(self, x, y):
        z = x - y
        z[z == -1] = 0
        return z

    def mul(self, x, y):
        z = np.bitwise_xor(x, y)
        return z

    def __dist__(self, x, y):
        z = np.bitwise_xor(x, y)
        return np.sum(z[z == 1]) / float(len(z))


class BinarySparseVector(AbstractVector):
    """
    A class representing a binary sparse Vector.
    """

    PERMUTATION_FACTOR = 8
    SPARSITY = 0.2

    def __init__(self, size, sparsity=SPARSITY):
        super().__init__(size)
        self.rep = AbstractVector.BINARY_SPARSE_VECTOR_TYPE

        # noinspection PyTypeChecker
        self.value = np.random.choice([0, 1], size=size, p=[1 - sparsity, sparsity])

    def add(self, x, y):
        # bundling in BSD is nothing but a fancy binding,
        # role-filler scheme - use same code
        z0 = self.__bitwise_or(x, y)

        # permutation factor
        k = 8

        # noinspection PyUnresolvedReferences
        return self.__permutate(k, x, z0)

    def sub(self, x, y):
        # TODO - add support for subtraction
        pass

    def mul(self, x, y):
        z0 = self.__bitwise_or(x, y)

        # permutation factor
        k = self.PERMUTATION_FACTOR

        # noinspection PyUnresolvedReferences
        return self.__permutate(k, x, z0)

    def __permutate(self, k, x, z0):
        zk = np.zeros((k, x.shape[0]), dtype=int)
        for i in range(0, k):
            zk[i] = np.random.permutation(z0)
        z = np.bitwise_or.reduce(zk)
        return np.bitwise_and(z, z0)

    def __bitwise_or(self, x, y):
        """
        Bitwise or using Numpy.
        :param x: array
        :param y: array
        :return: array
        """
        return np.bitwise_or(x, y)

    def __dist__(self, x, y):
        # noinspection PyTypeChecker
        d = 1 - np.sum(np.bitwise_and(x, y)) / np.sqrt(np.sum(x) * np.sum(y))
        return d


class BipolarVector(AbstractVector):
    """
    A class representing a bipolar Vector.
    """

    def __init__(self, size):
        super().__init__(size)
        self.rep = AbstractVector.BIPOLAR_VECTOR_TYPE
        # noinspection PyTypeChecker
        self.value = np.random.choice([-1.0, 1.0], size=size)

    def add(self, x, y):
        z = x + y
        z[z > 1] = 1.0
        z[z < -1] = -1.0
        z[z == 0] = np.random.choice([-1.0, 1.0], size=len(z[z == 0]))
        return z

    def sub(self, x, y):
        # TODO - add support for subtraction
        pass

    def mul(self, x, y):
        return x * y

    def __dist__(self, x, y):
        # noinspection PyTypeChecker
        return (len(x) - np.dot(x, y)) / (2 * float(len(x)))
