"""
This file creates classification reports for baseline_2 results only. For other uses, a parsing implmentation of results.jsons confusion_matrix field is required.

Hardcoded confusion matrices to feed following functions, these are results from tests (which aren't run anymore so this is fine)
Labels from 0 to N-1 are: 
    "AdenomeraAndre", "Ameeregatrivittata", "AdenomeraHylaedactylus", "HylaMinuta", "HypsiboasCinerascens", "HypsiboasCordobae", "LeptodactylusFuscus", "OsteocephalusOophagus", "Rhinellagranulosa", "ScinaxRuber"

These are acummulated confusion matrices over 10 folds.

kNN
662,0,0,7,0,1,0,1,1,0
1,541,0,2,0,0,0,0,0,0
0,0,3473,6,0,3,0,0,1,0
3,0,1,294,0,1,1,0,0,0
3,0,0,0,462,9,2,6,0,1
2,1,3,0,3,1102,1,4,0,0
0,0,0,0,2,2,266,1,2,0
1,0,0,0,5,3,0,102,0,0
0,0,0,1,0,0,0,0,64,1
0,0,1,0,0,0,0,0,0,146

ANN
666,0,0,6,1,1,0,1,1,1
2,540,0,2,0,0,0,0,0,0
0,0,3470,4,0,2,1,0,1,0
1,2,5,298,1,2,1,0,0,0
0,0,0,0,462,3,1,3,0,0
1,0,2,0,4,1107,1,0,0,1
0,0,0,0,0,1,265,0,2,1
1,0,0,0,4,3,1,110,0,0
0,0,0,0,0,1,0,0,64,1
1,0,1,0,0,1,0,0,0,144

"""

from sklearn.metrics import classification_report
import pandas as pd

def flatten_dataframe_to_labels(df):
    
    predicted_labels = []
    target_labels = []
    
    for row in df.index:
        for col in df.columns:
            v = df.at[row, col]
            predicted_instances = [row] * v
            target_instances = [col] * v

            predicted_labels += predicted_instances
            target_labels += target_instances

    return predicted_labels, target_labels

def generate_classification_report(Y_pred, Y_true, *args, **kwargs):

    return classification_report(Y_true, Y_pred, **kwargs)

if __name__ == "__main__":

    species = ["AdenomeraAndre", "Ameeregatrivittata", "AdenomeraHylaedactylus", "HylaMinuta", "HypsiboasCinerascens", "HypsiboasCordobae", "LeptodactylusFuscus", "OsteocephalusOophagus", "Rhinellagranulosa", "ScinaxRuber"]
    species_labels = [i for i in range(len(species))]
    #kwargs for sklearn.metric.classification_report-functions
    report_settings = {
        'target_names' : species,
        'digits' : 4,
        'output_dict' : False
        }
    
    knn_df = pd.DataFrame([
        [662,0,0,7,0,1,0,1,1,0],
        [1,541,0,2,0,0,0,0,0,0],
        [0,0,3473,6,0,3,0,0,1,0],
        [3,0,1,294,0,1,1,0,0,0],
        [3,0,0,0,462,9,2,6,0,1],
        [2,1,3,0,3,1102,1,4,0,0],
        [0,0,0,0,2,2,266,1,2,0],
        [1,0,0,0,5,3,0,102,0,0],
        [0,0,0,1,0,0,0,0,64,1],
        [0,0,1,0,0,0,0,0,0,146]
        ], index=species_labels, columns=species_labels)

    ann_df = pd.DataFrame([
        [666,0,0,6,1,1,0,1,1,1],
        [2,540,0,2,0,0,0,0,0,0],
        [0,0,3470,4,0,2,1,0,1,0],
        [1,2,5,298,1,2,1,0,0,0],
        [0,0,0,0,462,3,1,3,0,0],
        [1,0,2,0,4,1107,1,0,0,1],
        [0,0,0,0,0,1,265,0,2,1],
        [1,0,0,0,4,3,1,110,0,0],
        [0,0,0,0,0,1,0,0,64,1],
        [1,0,1,0,0,1,0,0,0,144]
        ], index=species_labels, columns=species_labels)

    dataframes = {'1NN' : knn_df, 'ANN' : ann_df}

    for type, df in dataframes.items():
        report = generate_classification_report(*flatten_dataframe_to_labels(df), **report_settings)
        if isinstance(report, dict):
            report = pd.DataFrame(report).transpose()
        
        print('')
        print(f'Report for dataframe {type}:')
        print(report)

        #to export dataframe as .tex table, import pathlib, open filehandle for each report and call report.to_latex(filehandle)