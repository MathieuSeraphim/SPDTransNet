import signal
import time
import optuna
from sqlalchemy.engine import Engine
from sqlalchemy import event
from jsonargparse import Namespace
from typing import Dict, Any, Union
from optuna.trial import Trial
from optuna.integration import PyTorchLightningPruningCallback
from _3_data_management._3_2_data_modules.DataModuleWrapper import get_datamodule_from_config_dict
from _4_models.ModelWrapper import get_model_from_config_dict
from _5_execution.TrainerWrapper import get_trainer_from_config_dict
from _5_execution._5_2_optuna.utils import before_class_instantiation, add_callback_to_trainer_config_dict, \
    optuna_standalone_execution, optuna_execution_on_previously_obtained_hparams
from _5_execution.run_model import run_model


class SignalException(Exception):
    pass


def interruption(signal, context):
    print('Received signal %d' % signal)
    raise SignalException


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
    except:
        time.sleep(1)
        set_sqlite_pragma(dbapi_connection, connection_record)


def optuna_run_model(trial: Trial, trainer_dict: Dict[str, Any], model_dict: Dict[str, Any],
                     datamodule_dict: Dict[str, Any], monitor: str, hparam_selection_files: Namespace,
                     random_seed: Union[int, None] = 42):

    print("Trial ID:", trial.number)

    pruning_callback = PyTorchLightningPruningCallback(trial, monitor=monitor)
    trainer_dict = add_callback_to_trainer_config_dict(trainer_dict, pruning_callback)
    trainer = get_trainer_from_config_dict(trainer_dict)

    model_dict, datamodule_dict = before_class_instantiation(trial, model_dict, datamodule_dict, hparam_selection_files)
    model = get_model_from_config_dict(model_dict)
    datamodule = get_datamodule_from_config_dict(datamodule_dict)

    run_model("fit", trainer, model, datamodule, random_seed)

    return trainer.checkpoint_callback.best_model_score.item()


# Note that the use of the validation MF1 score for Optuna monitoring is hardcoded
def optuna_execution(optuna_config: Namespace, execution_method: str, execution_kwargs: Dict[str, Any],
                     random_seed: Union[int, None] = 42):

    # Use this to interrupt the signal at a given time, if needed.
    signal.signal(signal.SIGUSR2, interruption)
    monitor = "mf1/validation"

    if execution_method == "standalone":
        trainer_dict, model_dict, datamodule_dict = optuna_standalone_execution(**execution_kwargs)
    elif execution_method == "from_hparams":
        trainer_dict, model_dict, datamodule_dict = optuna_execution_on_previously_obtained_hparams(**execution_kwargs)
    else:
        raise NotImplementedError

    # If True, there's no point in running a hyperparameter research
    assert not (optuna_config.hparam_selection_config.model is None and optuna_config.hparam_selection_config.datamodule is None)

    try:

        storage = optuna.storages.RDBStorage(
            url=optuna_config.storage,
            engine_kwargs={"connect_args": {"timeout": 1000}},
        )

        study = optuna.load_study(
            study_name=optuna_config.study_name,
            storage=storage,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=optuna_config.pruner.n_startup_trials,  # Minimal number of trials to run before pruning
                n_warmup_steps=optuna_config.pruner.n_warmup_steps,  # Number of network epochs to wait before pruning
                interval_steps=optuna_config.pruner.interval_steps  # Number of network epochs between pruner acts
            )
        )

        study.optimize(
            lambda trial: optuna_run_model(trial, trainer_dict, model_dict, datamodule_dict, monitor,
                                           optuna_config.hparam_selection_config, random_seed),
            n_trials=1)

    except SignalException:
        print("Program interrupted by signal.")


    
