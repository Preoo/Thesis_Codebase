#hyperparameters
hyperparams = {
    'epochs' : 30, #160
    'learning_rate' : 1e-3, #seems high, maybe cause dead units with ReLU
    'w_decay' : 1e-5, #weight decay hyperparam
    'batch_size' : 400,
    'report_interval' : 500,
    'use_stratified' : True,
    'n_folds' : 10,
    'run_kfolds' : True,
    'verbose' : False,
    'input_nodes':22,
    'output_nodes':10,
    'hidden_nodes':120,
    'hidden_layers':1,
    'dropout_prop':0.0,
    'use_groupnorm':True,
    'k_nearest' : 1,
    'knn_metric' : 'cosine', #supported values are: 'euclidian' | 'cosine'
    'loss_function' : 'crossentropy' #supported values are: 'crossentropy' | 'nllr'
}