from experiment_manager import cfg

# Retrivign the hyperparameters
hyperparams = cfg.config()
hyperparams.create_hp('learning_rate', 0.1, argparse=True)
hyperparams.create_hp('batch_size', 2, argparse=True)
hyperparams.create_hp('enable_conv', True, argparse=True)
hyperparams.parse_args()

# hyperparams.to_yml(file_path='logs/test.yml')
hyperparams.load_yml('logs/test.yml')
print(hyperparams)
print('hello')

