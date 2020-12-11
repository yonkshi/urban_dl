import optuna

from functools import partial

import damage_train
import experiment_manager.args

def objective(trial, cfg):
    cfg.TRAINER.LR = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    cfg.TRAINER.WD = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    cfg.TRAINER.B1 = 1.0 - trial.suggest_loguniform('one_minus_b1', 1e-4, 1.0)
    cfg.TRAINER.B2 = 1.0 - trial.suggest_loguniform('one_minus_b2', 1e-4, 1.0)
    cfg.MODEL.LOSS_TYPE = trial.suggest_categorical('loss_type', ['ComboLoss'])
    cfg.MODEL.BACKBONE.TYPE = trial.suggest_categorical('backbone', ['resnet34'])
    cfg.MODEL.BACKBONE.PRETRAINED = trial.suggest_categorical('pretrain', [False])
    cfg.AUGMENTATION.CROP_TYPE = trial.suggest_categorical('crop_type', ['importance'])

    cfg.OPTUNA.TRIAL_NUM = trial.number

    return damage_train.damage_train(trial, cfg)

def hyperparameter_search_argument_parser():
    parser = experiment_manager.args.default_argument_parser()
    parser.add_argument('--job-id', dest='job_id', type=str,
                        default='', help='Job ID (encompassing multiple trials) for naming runs')
    parser.add_argument('--trial-number', dest='trial_num', type=str,
                        default='', help='Number of the current trial for naming runs')
    return parser

def setup(args):
    cfg = damage_train.setup(args)
    cfg.JOB_ID = args.job_id
    cfg.TRIAL_NUM = args.trial_num
    return cfg

def main():
    parser = hyperparameter_search_argument_parser()
    args = parser.parse_known_args()[0]
    cfg = setup(args)

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(multivariate=True),
                                pruner=optuna.pruners.HyperbandPruner(
                                    min_resource=cfg.OPTUNA.MIN_RESOURCE,
                                    max_resource=cfg.OPTUNA.MAX_RESOURCE,
                                    reduction_factor=cfg.OPTUNA.REDUCTION_FACTOR
                                ),
                                study_name=cfg.NAME,
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
