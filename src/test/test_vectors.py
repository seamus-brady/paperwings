#  Copyright (c) 2025. Prediction By Invention https://predictionbyinvention.com/
#
#  THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import pickle  # nosec
import unittest
from pathlib import Path
from typing import Tuple

from src.paperwings.util.file_path_util import FilePathUtil
from src.paperwings.vector.vector import AbstractVector
from src.paperwings.vector.vector_space import VectorSpace


class TestVectors(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def get_test_vector_space(self, rep: str) -> Tuple:
        s = VectorSpace(rep=rep)

        u = s.add_vector("USA")
        d = s.add_vector("DOLLAR")
        m = s.add_vector("MEXICO")
        p = s.add_vector("PESOS")
        x = s.add_vector("COUNTRY")
        y = s.add_vector("CURRENCY")

        a = x * u + y * d
        b = x * m + y * p

        s.insert_vector(a, "USA Currency")
        s.insert_vector(b, "Mexican Currency")

        return s, a, b

    def test_vector_space_init(self) -> None:
        s, _, _ = self.get_test_vector_space(rep=AbstractVector.BINARY_VECTOR_TYPE)
        space_str = s.__str__()
        self.assertTrue(len(space_str) == 16335)
        self.assertTrue(s.size == 1000)

    def test_vector_space_rep(self) -> None:
        s = VectorSpace(rep="TEST")
        self.assertTrue(s.rep == "TEST")

    def test_vector_space_size(self) -> None:
        s = VectorSpace(size=5)
        self.assertTrue(s.size == 5)

    def test_binary_vector(self) -> None:
        s, a, b = self.get_test_vector_space(rep=AbstractVector.BINARY_VECTOR_TYPE)
        t1 = s.find_vector(a)
        t2 = s.find_vector(b)
        self.assertTrue(t1[0] == "USA Currency")
        self.assertTrue(t2[0] == "Mexican Currency")

    def test_del_vector(self) -> None:
        s, _, _ = self.get_test_vector_space(rep=AbstractVector.BINARY_VECTOR_TYPE)
        t = s.add_vector("test_me")
        self.assertTrue("test_me" == s.find_vector(t)[0])
        s.delete_vector("test_me")
        self.assertFalse("test_me" == s.find_vector(t)[0])

    def test_forget(self) -> None:
        s, _, _ = self.get_test_vector_space(rep=AbstractVector.BINARY_VECTOR_TYPE)
        t = s.add_vector("test_me")
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        s.decay()
        self.assertFalse("test_me" == s.find_vector(t)[0])

    def test_binary_sparse_vector(self) -> None:
        s, a, b = self.get_test_vector_space(
            rep=AbstractVector.BINARY_SPARSE_VECTOR_TYPE
        )
        t1 = s.find_vector(a)
        t2 = s.find_vector(b)
        self.assertTrue(t1[0] == "USA Currency")
        self.assertTrue(t2[0] == "Mexican Currency")

    def test_bipolar_vector(self) -> None:
        s, a, b = self.get_test_vector_space(rep=AbstractVector.BIPOLAR_VECTOR_TYPE)
        t1 = s.find_vector(a)
        t2 = s.find_vector(b)
        self.assertTrue(t1[0] == "USA Currency")
        self.assertTrue(t2[0] == "Mexican Currency")

    def test_binary_vector_bind0(self) -> None:
        s = VectorSpace(rep=AbstractVector.BINARY_VECTOR_TYPE)
        vec0 = s.add_vector("vec0")
        vec1 = s.add_vector("vec1")
        # add some dummy vectors
        for i in range(0, 100):
            s.add_vector()
        vec0_bind_vec1 = vec0 * vec1
        unbind = vec0_bind_vec1 * vec0
        self.assertTrue("vec1" == s.find_vector(unbind)[0])

    def test_binary_vector_bind1(self) -> None:
        s = VectorSpace(rep=AbstractVector.BINARY_VECTOR_TYPE)
        vec0 = s.add_vector("vec0")
        vec1 = s.add_vector("vec1")
        # add some dummy vectors
        for i in range(0, 100):
            s.add_vector()
        vec0_bind_vec1 = vec0 * vec1
        unbind = vec0_bind_vec1 * vec1
        self.assertTrue("vec0" == s.find_vector(unbind)[0])

    def test_subtract(self) -> None:
        s = VectorSpace(rep=AbstractVector.BINARY_VECTOR_TYPE)
        vec0 = s.add_vector("vec0")
        vec1 = s.add_vector("vec1")
        # add some dummy vectors
        for i in range(0, 100):
            s.add_vector()
        vec_added = vec0 + vec1
        vec_subtracted = vec_added - vec1
        self.assertTrue("vec0" == s.find_vector(vec_subtracted)[0])

    def test_vector_space_pickle(self) -> None:
        # create and save a vecor space
        vs_path: Path = Path(f"{FilePathUtil.storage_path()}vector_space.pkl")
        s, _, _ = self.get_test_vector_space(rep=AbstractVector.BINARY_VECTOR_TYPE)
        vec0 = s.add_vector("vec0")
        vec1 = s.add_vector("vec1")
        # add some dummy vectors
        for i in range(0, 100):
            s.add_vector()
        vec_added = vec0 + vec1
        vec_subtracted = vec_added - vec1
        self.assertTrue("vec0" == s.find_vector(vec_subtracted)[0])
        with vs_path.open("wb") as fp:
            pickle.dump(s, fp)

        # open a vector space and read
        with open(vs_path, "rb") as f:
            s2 = pickle.load(f)  # nosec

        self.assertIsNotNone(s2)
        self.assertTrue("vec0" == s2.find_vector(vec_subtracted)[0])


if __name__ == "__main__":
    unittest.main()
