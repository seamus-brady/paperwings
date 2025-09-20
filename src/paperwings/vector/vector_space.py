# MIT License
#
# Based on original code from https://github.com/mansourkheffache/hdc by Mansour Kheffache.
# Modifications have been made by Seamus Brady in 2025.
#
# Copyright (c) 2018 Mansour Kheffache
# Copyright (c) 2025 seamus@corvideon.ie
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

import math
import os
import random
import sqlite3
from typing import Dict, Optional, Tuple

from src.paperwings.exceptions.memory_exception import MemoryException
from src.paperwings.util.file_path_util import FilePathUtil
from src.paperwings.util.logging_util import LoggingUtil
from src.paperwings.vector.vector import AbstractVector

random.seed()


class VectorSpace:
    """
    A class for managing a collection of HD Vectors.
    """

    LOGGER = LoggingUtil.instance("<VectorSpace>")

    SQL_CREATE_VECTOR_TABLE = """
        CREATE TABLE IF NOT EXISTS vectors (
            name TEXT PRIMARY KEY,
            rep TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """

    SQL_INSERT_VECTOR = """INSERT OR IGNORE INTO vectors (name, rep) VALUES (?, ?)"""

    def __init__(
        self,
        size: int = 1000,
        rep: str = AbstractVector.BINARY_VECTOR_TYPE,
        storage_path: str = FilePathUtil.storage_path(),
        prefix: str = "default",
    ) -> None:
        """
        Init the VectorSpace.
        :param size: int, default size of each vector
        :param rep: str, vector representation.
        """

        try:
            self.size: int = size
            self.rep: str = rep
            self.vectors: Dict = {}
            self.storage_path = storage_path
            self.db_path = os.path.join(storage_path, f"{prefix}_vector_metadata.db")
            self.npz_path = os.path.join(storage_path, f"{prefix}_vector_store.npz")
            os.makedirs(self.storage_path, exist_ok=True)
            self._init_db()
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def _init_db(self) -> None:
        """Initialize the SQLite database for metadata storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(self.SQL_CREATE_VECTOR_TABLE)
            conn.commit()
            conn.close()
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def _random_name(self) -> str:
        """
        Return a random name for a vector.
        :return: str
        """

        try:
            return "".join(
                random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # nosec
                for i in range(8)  # nosec
            )  # nosec
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def __repr__(self) -> str:
        """
        VectorSpace to string.
        :return: str
        """

        try:
            return "".join("'%s' , %s\n" % (v, self.vectors[v]) for v in self.vectors)
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def __getitem__(self, x) -> AbstractVector:
        """
        Return one vector.
        :param x: str name
        :return: array
        """

        try:
            return self.vectors[x]
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def delete_vector(self, key) -> None:
        """
        Delete a vector from the VectorSpace.
        :param key:
        :return: None
        """

        try:
            del self.vectors[key]
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def add_vector(self, name: Optional[str] = None) -> AbstractVector:
        """
        Add a vector to the VectorSpace.
        :param name: str
        :return: array
        """

        try:
            if name is None:
                name = self._random_name()

            v = AbstractVector.new_vector(self.size, self.rep)

            self.vectors[name] = v

            # add to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(self.SQL_INSERT_VECTOR, (name, self.rep))
            conn.commit()
            conn.close()

            return v
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def insert_vector(self, v: AbstractVector, name: Optional[str] = None) -> str:
        """
        Insert a vector into the VectorSpace.
        :param v: array
        :param name: str
        :return: str
        """

        try:
            if name is None:
                name = self._random_name()
            self.vectors[name] = v
            return name
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def find_vector(self, x) -> Tuple[AbstractVector, float]:
        """
        Find the closest vector in distance terms in the VectorSpace.
        :param x: array
        :return: array
        """

        try:
            d: float = 1.0
            match: Optional[AbstractVector] = None

            for v in self.vectors:
                if self.vectors[v].dist(x) < d:
                    match = v
                    d = self.vectors[v].dist(x)

            return match, d  # type: ignore
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def exponential_decay(self, t, initial_value, decay_rate):
        """
        Calculates the remaining memory strength using exponential decay.

        Parameters:
        - t (float): Time elapsed.
        - initial_value (float): The initial strength or value of the memory.
        - decay_rate (float): The rate at which the memory decays.

        Returns:
        - float: The decayed memory strength.
        """

        try:
            return initial_value * math.exp(-decay_rate * t)
        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))

    def decay(self, decay_rate: float = 0.05, time_passed: int = 5) -> None:
        """
        Forget memories using exponential decay.
        The memory strength will decrease rapidly initially and slow down as time progresses.
        It's suitable when newer memories need to be significantly more potent than older ones quickly.
        """

        try:
            # Adjust the decay rate based on your needs
            memory_decay_rate = decay_rate

            # Time units since the memory was formed
            memory_time_passed = time_passed

            read_only_vectors = dict(self.vectors)  # copy

            for k, v in read_only_vectors.items():
                initial_memory_strength = v.strength
                v.strength = self.exponential_decay(
                    memory_time_passed, initial_memory_strength, memory_decay_rate
                )
                if v.strength <= 0.5:
                    self.delete_vector(k)

        except Exception as error:
            self.LOGGER.error(str(error))
            raise MemoryException(str(error))
