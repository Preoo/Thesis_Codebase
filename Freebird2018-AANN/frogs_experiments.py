import frogs_kNN as kNN
import frogs_ANN as ANN

from frogs_plotter import *

from frogs_hyperparams import hyperparams
from pathlib import Path
import json

class Frogs_ExperimentCase:
    def __init__(self, model, hyperparams, *args, description='default'):
        self.description = description
        self.model = model
        self.hyperparams = hyperparams

    def __repr__(self):
        return f"Testcase for {self.description}."

    def run_case(self):
        print(f"Executing testcase# {self.description}.")
        res = self.model.run(**self.hyperparams)
        res['description'] = self.description
        res['hyperparams'] = self.hyperparams
        return res

class Frogs_ExperimentsRunner:
    """Runs specified experiments and plots results (should save results as human readable json aswell in same dir) """
    def __init__(self, *args, **kwargs):
        
        self.modules = {
            'knn':kNN,
            'ann':ANN
            }
        self.default_params = kwargs.pop('default_params', hyperparams)

    def generate_param_range(self, for_param, from_value, to_value, with_step):
        import numpy as np
        N = to_value / with_step + 1
        return [{for_param:val} for val in np.linspace(from_value, to_value, num=N, dtype=type(to_value))]
        #range(from_value, to_value + with_step, with_step)

    def baseline_results(self):
        baseline_knn = Frogs_ExperimentCase(kNN, hyperparams, description="1NN").run_case()
        baseline_ann = Frogs_ExperimentCase(ANN, hyperparams, description="22L-120N-10-0.0dropout ANN").run_case()

        plotter = Frogs_Plotter()
        plot_settings = {
            'experiment_name' : 'Baseline',
            'plot_lineplots' : False,
            'plot_runtime' : True,
            'plot_cfmatrix' : True,
            'plot_boxplot' : True
            }

        plotter.plot_results(baseline_knn, baseline_ann, settings=plot_settings, dry_run=False)

    def test3(self, static_test_params={}, dynamic_test_params={}, test_label='default'):
        module = static_test_params.pop('module', 'knn')
        
        if dynamic_test_params:
            p = dynamic_test_params.get('p')
            p_to = dynamic_test_params.get('p_to')
            p_step = dynamic_test_params.get('p_step')
            settings = self.generate_param_range(p, self.default_params.get(p, 1), p_to, p_step)

        else:
            settings = [{'__testinfo__':'static test'}] #single test case with no dynamic params
            p = ''
        return [Frogs_ExperimentCase(self.modules[module], {**self.default_params, **static_test_params, **setting}, description=f"{setting.get(p, test_label)} {dynamic_test_params.get('var_tag', test_label)}").run_case() 
                for setting 
                in settings]

if __name__ == "__main__":

    runner = Frogs_ExperimentsRunner(default_params=hyperparams)
    #runner.baseline_results()
    plotter = Frogs_Plotter()

    plot_settings = {
        'experiment_name' : 'ANN-dropoutGN-200epochs',
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
    dyna_test_settings = {
        'p': 'dropout_prop',
        'p_to': 0.5,
        'p_step' : 0.1,
        'var_tag' : 'dropout'
        }

    results = [
        #runner.test3(static_test_params={'module' : 'ann', 'loss_function' : 'crossentropy'}, dynamic_test_params=dyna_test_settings)
        #Frogs_ExperimentCase(ANN, {**hyperparams, 'hidden_nodes':h, 'hidden_layers':l}, description=f"H={h},\nL={l}").run_case() for h in [20, 80, 150] for l in [1,2,3]
        Frogs_ExperimentCase(ANN, {**hyperparams, 'dropout_prop':0.0, 'use_groupnorm':True}, description=f"Batch Norm").run_case()
        ]

    #Runs plotter to generate figures and plots. Use dry_run to avoid saving to a dir when developing
    plotter.plot_results(*results, settings=plot_settings, dry_run=True)




    #runner = Frogs_ExperimentsRunner(name='test', module='knn', default_params=hyperparams)
    #print('-' * 80)
    #test_results = runner.test3()
    '''
    settings = {
                'plot_cfmatrix' : True,
                'plot_boxplot' : True,
                'plot_runtime' : True,
                'plot_lineplots' : True,
                'lineplot_keys' : (['hyperparams','k_nearest'],['Accuracy']),
                'lineplot_legends' : [],
                'lineplot_labels' : {'xlabel':'x-default', 'ylabel':'y-default'}
                }
    '''

    #with open(Path.cwd() / 'Tests' / 'test_results_for_plotter.json', mode='r') as f:
    #    test_results = json.load(f)
    #    #json.dump(test_results, f)

    #from random import uniform
    #modded_results = [{**r} for r in test_results]
    #for r in modded_results:
    #    rnd_mod = uniform(-0.5, 0.5)
    #    r['Runtime'] += rnd_mod
    #plotter = Frogs_LinePlot(xlabel='kNN', ylabel='Tarkkuus')
    #plotter = Frogs_BarPlot()
    #plotter.plot(test_results, modded_results, legend_labels=['Test', 'Modded'])
    #plotter.plot(test_results, legend_labels=['Sample', 'Modded'])
    #plotter.save_plots_to_file(Path.cwd() / 'Results')
