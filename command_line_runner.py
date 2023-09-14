from jsonargparse import ArgumentParser, namespace_to_dict
from _5_execution._5_2_optuna.optuna_run_model import optuna_execution
from _5_execution.run_model import standalone_execution, execution_on_previously_obtained_hparams

execution_methods = ["standalone", "from_hparams"]


# Note that the use of the validation MF1 score for Optuna monitoring is hardcoded
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--execution_method", type=str, default="standalone")

    parser.add_argument("--execution_type", type=str, default="fit")
    parser.add_argument("--global_seed", type=int, default=None)

    parser.add_argument("--trainer_config_file", type=str, default=None)
    parser.add_argument("--trainer_config.logger_logs_location", type=str, default=".")
    parser.add_argument("--trainer_config.logger_logs_folder_name", type=str, default="lightning_logs")
    parser.add_argument("--trainer_config.logger_version", type=int, default=None)

    parser.add_argument("--model_config_file", type=str, default=None)
    parser.add_argument("--model_config.cross_validation_fold_index", type=str, default=None)

    parser.add_argument("--datamodule_config_file", type=str, default=None)
    parser.add_argument("--datamodule_config.batch_size", type=int, default=None)
    parser.add_argument("--datamodule_config.cross_validation_fold_index", type=int, default=None)

    parser.add_argument("--hparams_config_file", type=str, default=None)

    parser.add_argument("--optuna_flag", action="store_true")

    parser.add_argument("--optuna.study_name", type=str, default='dev')
    parser.add_argument("--optuna.storage", type=str, default='sqlite:///db/database.db')

    parser.add_argument("--optuna.pruner.n_startup_trials", type=int, default=10)  # Minimal number of trials to run before pruning
    parser.add_argument("--optuna.pruner.n_warmup_steps", type=int, default=3)  # Number of network epochs to wait before pruning
    parser.add_argument("--optuna.pruner.interval_steps", type=int, default=1)  # Number of network epochs between pruner acts

    parser.add_argument("--optuna.hparam_selection_config.model", type=str, default=None)
    parser.add_argument("--optuna.hparam_selection_config.datamodule", type=str, default=None)

    command_line_inputs = parser.parse_args()

    trainer_config_file = command_line_inputs.trainer_config_file
    model_config_file = command_line_inputs.model_config_file
    datamodule_config_file = command_line_inputs.datamodule_config_file
    hparams_config_file = command_line_inputs.hparams_config_file

    execution_type = command_line_inputs.execution_type
    global_seed = command_line_inputs.global_seed

    trainer_config_modifications_as_dict = namespace_to_dict(command_line_inputs.trainer_config)
    model_config_modifications_as_dict = namespace_to_dict(command_line_inputs.model_config)
    datamodule_config_modifications_as_dict = namespace_to_dict(command_line_inputs.datamodule_config)

    execution_method = command_line_inputs.execution_method
    assert execution_method in execution_methods

    run_with_optuna_flag = command_line_inputs.optuna_flag
    optuna_config = command_line_inputs.optuna

    if run_with_optuna_flag:
        assert execution_type == "fit"

        if execution_method == "standalone":
            execution_kwargs = {
                "trainer_config_file": trainer_config_file,
                "trainer_config_kwargs": trainer_config_modifications_as_dict,
                "model_config_file": model_config_file,
                "model_config_kwargs": model_config_modifications_as_dict,
                "datamodule_config_file": datamodule_config_file,
                "datamodule_config_kwargs": datamodule_config_modifications_as_dict
            }
        elif execution_method == "from_hparams":
            execution_kwargs = {
                "trainer_config_file": trainer_config_file,
                "trainer_config_kwargs": trainer_config_modifications_as_dict,
                "model_config_kwargs": model_config_modifications_as_dict,
                "datamodule_config_kwargs": datamodule_config_modifications_as_dict,
                "hparams_file": hparams_config_file
            }
        else:
            raise NotImplementedError

        optuna_execution(optuna_config, execution_method, execution_kwargs, global_seed)


    else:
        if execution_method == "standalone":
            standalone_execution(execution_type, trainer_config_file, trainer_config_modifications_as_dict,
                                 model_config_file, model_config_modifications_as_dict, datamodule_config_file,
                                 datamodule_config_modifications_as_dict, global_seed)
        elif execution_method == "from_hparams":
            execution_on_previously_obtained_hparams(
                execution_type, trainer_config_file, trainer_config_modifications_as_dict,
                model_config_modifications_as_dict, datamodule_config_modifications_as_dict, hparams_config_file,
                global_seed)
        else:
            raise NotImplementedError


