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

    def test_ConfusionMatrixCumulative(self):

        cm1 = ConfusionMatrix(labels=["a","b"])
        cm2 = ConfusionMatrix(labels=["a","b"])

        cm1.results += 1
        cm2.results += 2
        self.assertTrue(np.array_equal(cm1.matrix, np.ones((2,2), dtype=np.int)), msg='Dataframe in confusionmatrix cant be added into')
        cm3 = cm1 + cm2
        expected = np.array([[3, 3], [3, 3]],dtype=np.int)
        self.assertTrue(np.array_equal(cm3.matrix, expected), msg='__add__ didnt yield expected result')
        self.assertTrue(np.array_equal(cm3.matrix, cm1.matrix), msg='__add__ didnt mutate first class')
        self.assertTrue(isinstance(cm3, ConfusionMatrix), msg='__add__ didnt return a ConfusionMatrix class')

    def test_kNN_resolves_correct_label(self):
        #dummy data x,y,z : xy are data and z is label
        
        mock_data = np.array([[1,1,1], [2,2,2], [3,3,2]], dtype=np.float)
        feat, label = mock_data[:, :2], mock_data[:, 2]
        
        from frogs_data import Frogs_Dataset
        X = Frogs_Dataset(frogs=feat[:2, :], labels=label[:2])
        y = Frogs_Dataset(frogs=feat[2:, :], labels=label[2:])

        #import torch

        Y, expected = y[0]
        from frogs_kNN import Frogs_kNN
        knn = Frogs_kNN()
        res = knn.fit(X, Y)

        was_correct = np.array_equal(expected, res)
        self.assertTrue(was_correct, msg='kNN returned wrong label')
if __name__ == '__main__':
    unittest.main()
