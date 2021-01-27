import optuna

import os
import time
from functools import partial

import damage_train
import experiment_manager.args

def objective(trial, cfg):
    cfg.TRAINER.LR = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    cfg.TRAINER.WD = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    # cfg.TRAINER.B1 = 1.0 - trial.suggest_loguniform('one_minus_b1', 1e-4, 1.0)
    # cfg.TRAINER.B2 = 1.0 - trial.suggest_loguniform('one_minus_b2', 1e-4, 1.0)
    # cfg.TRAINER.CE_CLASS_BALANCE.ENABLED = trial.suggest_categorical('ce_class_balance', [True])
    # cfg.MODEL.LOSS_TYPE = trial.suggest_categorical('loss_type', ['ComboLoss'])
    # cfg.MODEL.BACKBONE.TYPE = trial.suggest_categorical('backbone', ['resnet34'])
    # cfg.MODEL.BACKBONE.PRETRAINED = trial.suggest_categorical('pretrain', [True])
    # cfg.MODEL.SIAMESE.PRETRAINED = cfg.MODEL.BACKBONE.PRETRAINED
    # cfg.AUGMENTATION.CROP_TYPE = trial.suggest_categorical('crop_type', ['importance'])
    # cfg.AUGMENTATION.IMAGE_OVERSAMPLING_TYPE = trial.suggest_categorical('oversampling_type', ['per_class'])
    # cfg.MODEL.SIAMESE.ENABLED = trial.suggest_categorical('siamese', [True])
    # cfg.DATASETS.LOCALIZATION_MASK.ENABLED = trial.suggest_categorical('localization_mask', [False])
    cfg.OPTUNA.TRIAL_ID = trial.number

    return damage_train.damage_train(trial, cfg)

def hyperparameter_search_argument_parser():
    parser = experiment_manager.args.default_argument_parser()
    parser.add_argument('--job-id', dest='job_id', type=str,
                        default='', help='Job ID (encompassing multiple trials) for naming runs')
    parser.add_argument('--trial-number', dest='trial_number', type=str,
                        default='', help='Number of the current trial for naming runs')
    parser.add_argument('--wandb-project', dest='wandb_project', type=str,
                        default='urban_dl_ablation', help='Wandb project')
    return parser

def setup(args):
    cfg = damage_train.setup(args)
    cfg.JOB_ID = args.job_id
    cfg.TRIAL_NUM = args.trial_number
    name_list = ['job', cfg.JOB_ID, 'trial', cfg.TRIAL_NUM]
    cfg.NAME = '_'.join(name_list)
    cfg.TAGS += ['optuna']
    cfg.PROJECT = args.wandb_project

    if args.log_dir: # Override Output dir
        cfg.OUTPUT_DIR = os.path.join(args.log_dir, cfg.NAME)
    else:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_BASE_DIR, cfg.NAME)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    parser = hyperparameter_search_argument_parser()
    args = parser.parse_known_args()[0]
    cfg = setup(args)

    # Delay startup for other trials to have give the database some time to initialize
    if cfg.TRIAL_NUM and int(cfg.TRIAL_NUM) != 1:
        time.sleep(60)

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True),
                                pruner=optuna.pruners.HyperbandPruner(
                                    min_resource=cfg.OPTUNA.MIN_RESOURCE,
                                    max_resource=cfg.OPTUNA.MAX_RESOURCE,
                                    reduction_factor=cfg.OPTUNA.REDUCTION_FACTOR
                                ),
                                study_name=cfg.CONFIG,
                                storage=cfg.OPTUNA.DB_PATH,
                                direction='maximize',
                                load_if_exists=True)

    configured_objective = partial(objective, cfg=cfg)
    study.optimize(configured_objective, n_trials=1)

    print(f'Finished trial: {study.trials[-1]}')
    try:
        print(f'Best performing trial so far: {study.best_trial}')
    except ValueError:
        pass

if __name__ == '__main__':
    main()
