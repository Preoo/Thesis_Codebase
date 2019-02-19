import unittest
from frogs_utils import generate_datasets
import pandas as pd
import numpy as np
from pathlib import Path

class Test_frogs_utilsTest(unittest.TestCase):
    def setUp(self):

        testdata = Path(__file__).parent / 'testcase.csv'

        self.dummy_data = pd.read_csv(testdata)
        self.dummy_data["labels"] = pd.get_dummies(self.dummy_data.Species).values.tolist()
        print(self.dummy_data.describe())

    def test_A(self):
        
        frogs_all = self.dummy_data.to_numpy(copy=True)
        #print(frogs_all[0][26])

        #0:22, 24, 26

        frogs_features = frogs_all[:, 0:22]
        frogs_labels = frogs_all[:, 26]

        print(frogs_features)
        print(frogs_labels)

        self.assertTrue(True, msg='Test A is only to debug some behavior, always works.')

    def test_GenerateDataset(self):

        X, Y, species = generate_datasets(self.dummy_data)
        self.assertGreater(len(species), 0, msg='It should generate list of labels to recover species name from model')
        self.assertGreater(len(X), len(Y), msg='Datset should be split into two sets where train set is larger')

    def test_CanEnumerateOverDataset(self):

        X, Y, _ = generate_datasets(self.dummy_data)
        count = 0
        for _ in range(len(X)):
            count = count + 1
        self.assertEqual(count, len(X), msg='Was able to iterate over dataset in for range loop')
if __name__ == '__main__':
    unittest.main()
