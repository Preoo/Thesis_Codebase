from pathlib import Path
import json
from frogs_plotter import *

#load all results.json files from specified dir as dicts.
def load_json_to_dict(*files, dir='Results/ANN-L2reg-80epochs'):

    for filename in files:
        fp = Path.cwd() / dir / filename
        with open(fp) as json_file:
            try:
                yield json.load(json_file)
            except json.JSONDecodeError:
                print(f'Failed to decode file: {fp} !')
                raise

#sort such that it maintains numberical ordering
files_to_load = [
         'results_additional_1.json',
         'results_additional_2.json',
         'results_additional_3.json',
         'results_additional_4.json',
         'results_additional_5.json',
         'results.json'
         ]

results = load_json_to_dict(*files_to_load)

#for r in results:
#    print(r.get('description'))

#feed those to frogs_plotter to generate new graphs
plotter = Frogs_Plotter()
plot_settings = {
        'experiment_name' : 'ANN-L2reg-sorted-80epochs',
        'clip_epochs' : 10, #0 is evaluated as false during plotting
        'plot_cfmatrix' : False,
        'plot_boxplot' : True,
        'plot_runtime' : False,
        'plot_lineplots' : True,
        'plot_lineplots_from_training' : True,
        'lineplot_keys' : (['TrainingStats', 'epoch'], ['TrainingStats','training_eval']),
        'lineplot_legends' : [],
        'lineplot_labels' : {'xlabel':'epochs', 'ylabel':'Tarkkuus'}
        }


plotter.plot_results(*results, settings=plot_settings, dry_run=False)