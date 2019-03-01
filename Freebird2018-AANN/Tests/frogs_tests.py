import unittest
from frogs_utils import generate_datasets, ConfusionMatrix
import pandas as pd
import numpy as np
from pathlib import Path

#This is a project relating to master's thesis, not a software dev job so positive testing shall suffice
class Test_frogs_utilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        testdata = Path(__file__).parent / 'testcase.csv'

        cls.mock_df = pd.read_csv(testdata)
        cls.mock_df["labels"] = pd.get_dummies(cls.mock_df.Species).values.tolist()
        #print(self.dummy_data.describe())

    def test_GenerateDataset(self):

        X, Y, species = generate_datasets(self.mock_df)
        #print(species)
        self.assertGreater(len(species), 0, msg='It should generate list of labels to recover species name from model')
        self.assertGreater(len(X), len(Y), msg='Dataset should be split into two sets where train set is larger')

    def test_CanEnumerateOverDataset(self):

        X, Y, _ = generate_datasets(self.mock_df)
        count = 0
        for _ in range(len(X)):
            count = count + 1
        self.assertEqual(count, len(X), msg='Was not able to iterate over dataset in for range loop')

    def test_ConfusionMatrixHasCorrectShape(self):
        labels = ["TRUE", "FALSE"]
        expected = (len(labels), len(labels))
        cm = ConfusionMatrix(labels=labels)
        expected_matrix = np.zeros(expected, dtype=np.int)
        self.assertEqual(expected, cm.shape, msg='Confusionmatrix init failed, incorrect shape')
        self.assertTrue(np.array_equal(expected_matrix, cm.matrix), msg='Confusion matrix had incorrect values or datatypes')

    def test_ConfusionMatrixReturnsCorrectResult(self):
        labels = ["a", "b"]
        #labels indexes: a:0, b:1
        
        cm = ConfusionMatrix(labels=labels)
        y = np.array([0,0,1,1,1]).reshape((-1, 1)) #mock of predicted
        Y = np.array([0,0,0,1,1]).reshape((-1, 1)) #mock of target
        cm.add_batch(predicted_labels=y, target_labels=Y)
        ''' Expected confusion matrix

            a b
        a   2 0
        b   1 2
        '''
        expected = np.array([[2, 0], [1, 2]])
        self.assertTrue(np.array_equal(cm.matrix, expected), msg='Confusion matrix added batches wrong. See test for more information.')

if __name__ == '__main__':
    unittest.main()
