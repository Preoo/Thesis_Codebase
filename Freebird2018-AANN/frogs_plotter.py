import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.style.use('seaborn-paper')

class Frogs_Plotter:
    """
    Use matplotlib.pyplot library to draw graphs.
    Save graphs and results as json object to directory.
    """
    

    def plot_results(self, results, *additional_results, settings={}, dry_run=True, **kwargs):
        """ Generate plots, graphs and save accuracy statistics. """
        #plt.plot training loss over epochs, data is list of folds with list of dicts {epoch:n, loss:m}
        #plt.boxplot for eval stats
        #plt.matshow to render confusionamtrix
        if settings is None:
            settings = {
                'plot_cfmatrix' : True,
                'plot_boxplot' : True,
                'plot_runtime' : True,
                'plot_lineplots' : True,
                'lineplot_keys' : ([''],['']),
                'lineplot_legends' : ['default'],
                'lineplot_labels' : {'xlabel':'x-default', 'ylabel':'y-default'}
                }
        #Create corresponding class-instances for each plot type and plot results for each. Finally, call save on all plots. 
        plots = []
        results_dir = Path.cwd() / 'Results'
        if 'experiment_name' in settings:
            results_dir = results_dir / settings['experiment_name']

        #Create confusion matrix plots
        if settings.get('plot_cfmatrix', False):
            
           for res in (results, *additional_results):
               a = Frogs_CMatrixPlot(plot_prefix=res.get('description', ''))
               a.plot(res)
               plots.append(a)
            
            #plots.append(Frogs_CMatrixPlot(plot_prefix=results.get('description', '')).plot(results))

        #Create boxplots for all in results
        if settings.get('plot_boxplot', False):
            a = Frogs_BoxPlot(ylabel='Tarkkuus')
            a.plot(results, *additional_results)
            plots.append(a)

        #Create barplots for runtimes for first set of results
        if settings.get('plot_runtime', False):
            a = Frogs_BarPlot(ylabel='', xlabel='Testin suoritusaika (s)', plot_prefix='runtime')
            a.plot(results, *additional_results, legend_labels=settings['lineplot_legends'])
            plots.append(a)
        
        #Create lineplots for cases based on settings
        if settings.get('plot_lineplots', False):
            a = Frogs_LinePlot(**settings['lineplot_labels'])
            a.plot(results, *additional_results, with_keys=settings['lineplot_keys'], legend_labels=settings['lineplot_legends'], clip_epochs=settings.get('clip_epochs'))
            plots.append(a)

        #save all figures in experiment folder
        for plot in plots:
            plot.save_plots_to_file(results_dir, dry_run=dry_run)

        #save all results as json, just in case
        if not dry_run:
            import json
            from frogs_utils import ConfusionMatrix
            """ Encode custom ConfusionMatrix class as json-string """
            class ConfusionEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, ConfusionMatrix):
                        return repr(obj)
                    return super().default(obj)


            with open(results_dir / 'results.json', mode='wt') as f:    
                json.dump(results, f, cls=ConfusionEncoder)

            if additional_results:
                for i, r in enumerate(additional_results, 1):
                    file = results_dir / f'results_additional_{i}.json'
                    with open(file, mode='wt') as f:
                        json.dump(r, f, cls=ConfusionEncoder)



class Frogs_PlotBase:
    def __init__(self, *args, xlabel='', ylabel='', rows=1, cols=1, plot_prefix='', **kwargs):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.nrows = rows
        self.ncols = cols
        self.plot_file = 'base.pdf'
        self.plot_prefix = plot_prefix
        #Create empty figure and axes based on shape. Subclasses should operate on this.
        self.fig, self.ax = plt.subplots(self.nrows, self.ncols, sharex=True, sharey=True, **kwargs)

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    @property
    def figure(self):
        return self.fig
    
    def plot(self, data):
        raise NotImplementedError('You must override `plot` function. Baseclass function was called.')

    def save_plots_to_file(self, path, dry_run=True):

        if dry_run:
            plt.show()
        else:
            path.mkdir(parents=True, exist_ok=True)
            output = path / self.plot_file
            self.fig.savefig(output, bbox_inches='tight')

class Frogs_LinePlot(Frogs_PlotBase):
    """ Lineplot with optional type params to ease plotting loss, accuracy or something else. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_file = f'{self.plot_prefix}_line.pdf'
        
    def plot(self, *data, with_keys=(['hyperparams','k_nearest'], ['Accuracy']), legend_labels=None, clip_epochs=None, **kwargs):
        if not isinstance(data, tuple):
            data = (data,) #if results is single instance, just create a tuple from it

        x_keys, y_keys = with_keys

        def nested_get(from_dict, keys):
            #Assumptions: If we encounter a list, this is either last key or 2nd last key in which case calculate averages over folds for that key
            try:
                value = from_dict
                for key in keys:
                    if isinstance(value, list):
                        return calculate_folds_avg(value, col=key)
                    else:
                        value = value[key]
                return value

            except KeyError:
                print('Tried to retrive missing keys, skipping plotting')
                return []

        #'epoch', 'loss', 'training_eval'
        def calculate_folds_avg(folds, col='training_eval'):
            folds_array = np.array([f[col] for f in folds])
            return np.mean(folds_array, axis=0).tolist()
            
        for result in data:
            
            x = nested_get(result, x_keys)
            y = nested_get(result, y_keys)
            
            if clip_epochs:
                x = x[clip_epochs:]
                y = y[clip_epochs:]

            self.ax.plot(x, y)

        if legend_labels:
            self.ax.legend(legend_labels)
        else:
            self.ax.legend([result['description'] for result in data])

class Frogs_BoxPlot(Frogs_PlotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_file = f'{self.plot_prefix}_box.pdf'

    def plot(self, data, *args, **kwargs):

        #list of lists where inner list contains eval-scores.
        #outer list has results for all experiments.
        #boxplot expects a list of lists toplot over set of experiments.

        if args:
            data = [data, *args]

        if not isinstance(data, list):
            data = [data] #if results is single instance, just create a list from it

        

        self.ax.boxplot([[x for x in case['EvaluationStats']] for case in data])
        self.ax.set_xticklabels([case['description'] for case in data])

class Frogs_CMatrixPlot(Frogs_PlotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_file = f'{self.plot_prefix}_cfmatrix.pdf'

    def plot(self, data, *args, **kwargs):
        try:
            im = self.ax.imshow(data['ConfusionMatrix'].falsepositives, interpolation='none', aspect='auto', cmap='coolwarm')
            self.ax.set_xticks(range(len(data['Labels'])))
            self.ax.set_yticks(range(len(data['Labels'])))
            self.ax.set_xticklabels([f"{label:.5}.." for label in data['Labels']])
            self.ax.set_yticklabels(data['Labels'])
            plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(len(data['Labels'])):
                for j in range(len(data['Labels'])):
                    self.ax.text(j, i, data['ConfusionMatrix'].falsepositives[i, j], ha="center", va="center", color="w")
    
            cbar = self.fig.colorbar(ax=self.ax, mappable=im, orientation='vertical')
            cbar.set_label('Virheellisten luokituksien määrä')
        except:
            print('CFMatrixPlot.plot got results data without ConfusionMatrix-attribute OR list of results, which is not supported.')
            raise

class Frogs_BarPlot(Frogs_PlotBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_file = f'{self.plot_prefix}_bar.pdf'

    def plot(self, *args, **kwargs):

        runtimes = [res['Runtime'] for res in args]
        models = [res['description'] for res in args]
        self.ax.bar(models, runtimes)
