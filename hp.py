from experiment_manager import cfg

# Retrivign the hyperparameters
hyperparams = cfg.config()
hyperparams.create_hp('learning_rate', 0.1, argparse=True)
hyperparams.create_hp('batch_size', 2, argparse=True)
hyperparams.create_hp('enbable_conv', True, argparse=True)
hyperparams.parse_args()
print(hyperparams)