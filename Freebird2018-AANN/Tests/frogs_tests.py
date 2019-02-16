import unittest
from frogs_utils import generate_datasets
import pandas as pd
from pathlib import Path

class Test_frogs_utilsTest(unittest.TestCase):
    def setUp(self):

        testdata = Path("f:\Documents\Visual Studio 2017\Projects\Freebird2018-AANN\Freebird2018-AANN\Tests\testcase.csv")

        self.dummy_data = pd.read_csv(testdata)
        print(self.dummy_data)

    def test_A(self):

        print("Just works")
        self.fail(msg="Didn't work")

if __name__ == '__main__':
    unittest.main()
