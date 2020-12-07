import optuna

from functools import partial

import damage_train
import experiment_manager.args

def objective(trial, cfg):
    cfg.TRAINER.LR = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    cfg.TRAINER.WD = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    cfg.TRAINER.B1 = 1.0 - trial.suggest_loguniform('one_minus_b1', 1e-4, 1.0)
    cfg.TRAINER.B2 = 1.0 - trial.suggest_loguniform('one_minus_b2', 1e-4, 1.0)
    return damage_train.damage_train(trial, cfg)

def main():
    args = experiment_manager.args.default_argument_parser().parse_known_args()[0]
    cfg = damage_train.setup(args)

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

    trials = study.get_trials()

    print(f'Finished trial: {trials[-1]}')
    print(f'Best performing trial so far: {study.best_trial}')

if __name__ == '__main__':
    main()
