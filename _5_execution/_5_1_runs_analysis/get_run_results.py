import csv
import warnings
from os import scandir, mkdir, listdir
from os.path import join, abspath, basename, isfile, dirname, realpath, isdir
from typing import Union, Dict, List, Any
import yaml
from _3_data_management._3_2_data_modules.DataModuleWrapper import modify_datamodule_config_dict, \
    get_datamodule_from_config_dict
from _5_execution.TrainerWrapper import get_trainer_config_dict_from_file, get_trainer_from_config_dict
from _5_execution._5_1_runs_analysis.utils import get_hparams_to_track_dict, \
    get_best_run_stats_to_track_dict
from _5_execution.utils import get_model_from_checkpoint_file, get_execution_configs_folder_path
from _5_execution.run_model import get_model_and_datamodule_dicts_from_hparams_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from _4_models.BaseModel import BaseModel
from operator import itemgetter

valitation_set = getattr(BaseModel, "VALIDATION_SET_NAME")
test_set = getattr(BaseModel, "TEST_SET_NAME")


shortened_set_names_dict = {valitation_set: "Val", test_set: "Test"}
def get_stat_column_name(stat_name: str, set_name: str, obtained_by_rerunning: bool = False):
    assert set_name in [valitation_set, test_set]
    output_string = "%s %s" % (shortened_set_names_dict[set_name], stat_name)
    if obtained_by_rerunning:
        return "<" + output_string + ">"
    return output_string


supported_additional_operations_list = []
def get_run_results(hparams_to_track: Dict[str, List[str]], best_run_stats_to_track: Dict[str, List[str]],
                    monitor: str = "mf1", test_output_name: str = "", log_folders_location: str = ".",
                    lightning_log_folder_name: str = "lightning_logs",
                    skip_runs_with_mf1_under_value: Union[float, None] = None, run_on_test_set: bool = True,
                    rerun_on_validation_set: bool = False, trainer_config_file: Union[str, None] = None,
                    trainer_config_modifications_as_dict: Union[Dict[str, str], None] = None,
                    override_batch_size_with_value: Union[int, None] = None, save_confusion_matrices: bool = False,
                    list_of_additional_operations: Union[List[Dict], None] = None,
                    restart_at_run_name: Union[int, None] = None, ignore_prefix: Union[int, None] = None):

    run_skipping_threshold = 0
    if skip_runs_with_mf1_under_value is not None:
        assert 0 <= skip_runs_with_mf1_under_value < 1
        run_skipping_threshold = skip_runs_with_mf1_under_value

    if override_batch_size_with_value is not None:
        assert override_batch_size_with_value >= 1

    model_hparams_to_track = []
    if "model" in hparams_to_track.keys():
        model_hparams_to_track = hparams_to_track["model"]

    datamodule_hparams_to_track = []
    if "datamodule" in hparams_to_track.keys():
        datamodule_hparams_to_track = hparams_to_track["datamodule"]

    hparams_order = []
    if "order" in hparams_to_track.keys():
        hparams_order = hparams_to_track["order"]
        assert len(hparams_order) == len(model_hparams_to_track) + len(datamodule_hparams_to_track)
        for hparam in model_hparams_to_track:
            assert hparam in hparams_order and hparam not in datamodule_hparams_to_track
        for hparam in datamodule_hparams_to_track:
            assert hparam in hparams_order

    test_stats_to_track = None
    assert valitation_set in best_run_stats_to_track.keys()
    validation_stats_to_track = best_run_stats_to_track[valitation_set]
    if run_on_test_set:
        assert test_set in best_run_stats_to_track.keys()
        test_stats_to_track = best_run_stats_to_track[test_set]

    trainer_config_dict = None
    if rerun_on_validation_set or run_on_test_set:
        assert trainer_config_file is not None
        assert trainer_config_modifications_as_dict is not None
        trainer_config_dict = get_trainer_config_dict_from_file(trainer_config_file,
                                                                **trainer_config_modifications_as_dict)

    log_folders_location = abspath(log_folders_location)
    lightning_log_folder = join(log_folders_location, lightning_log_folder_name)
    runwise_lightning_log_subfolders = [f.path for f in scandir(lightning_log_folder) if f.is_dir()]
    runwise_lightning_log_subfolders.sort()

    current_script_directory = dirname(realpath(__file__))
    output_folder = join(current_script_directory, "output")
    if not isdir(output_folder):
        mkdir(output_folder)

    # If not empty, must start with an underscore
    if test_output_name != "" and test_output_name[0] != "_":
        test_output_name = "_" + test_output_name

    csv_output_filename = join(output_folder, "runs_analysis" + test_output_name + ".csv")
    write_first_line_flag = False
    if restart_at_run_name is None or not isfile(csv_output_filename):
        csv_output_file = open(csv_output_filename, "w", newline="")
        csv_output_file.close()
        write_first_line_flag = True

    confusion_matrices_folder = None
    if save_confusion_matrices:
        confusion_matrices_folder = join(output_folder, "confusion_matrices" + test_output_name)
        if not isdir(confusion_matrices_folder):
            mkdir(confusion_matrices_folder)

    for runwise_lightning_log_subfolder in runwise_lightning_log_subfolders:
        runwise_lightning_log_subfolder_split_name = basename(runwise_lightning_log_subfolder).split("_")
        if len(runwise_lightning_log_subfolder_split_name) == 2\
                and runwise_lightning_log_subfolder_split_name[-2] == "version":
            run_id = runwise_lightning_log_subfolder_split_name[-1]
        else:
            run_id = basename(runwise_lightning_log_subfolder)

        skip_flag = False
        if restart_at_run_name is not None:
            if run_id < str(restart_at_run_name):
                skip_flag = True
        if ignore_prefix is not None:
            prefix_str = str(ignore_prefix)
            if len(run_id) == len(prefix_str) + 2:
                if run_id[:len(prefix_str)] == prefix_str:
                    skip_flag = True
        if skip_flag:
            print("Skipping run %s." % run_id)
            continue

        run_dict = {"Run ID": run_id}

        print("\nProcessing run %s...\n" % run_id)

        # Get recorded validation stats from logger

        recorded_events_accumulator = EventAccumulator(runwise_lightning_log_subfolder)
        recorded_events_accumulator.Reload()

        try:
            monitor_validation_events = recorded_events_accumulator.Scalars("%s/%s" % (monitor, valitation_set))
        except KeyError as k:
            warnings.warn("Obtained the following KeyError: %s. Skipping run." % str(k))
            continue

        monitor_validation_data = [event_data.value for event_data in monitor_validation_events]
        best_epoch_index, monitor_validation_value = max(enumerate(monitor_validation_data), key=itemgetter(1))
        if monitor_validation_value < run_skipping_threshold:
            warnings.warn("Validation %s value of %4f, under threshold of %.4f. Skipping run."
                          % (monitor, monitor_validation_value, run_skipping_threshold))
            continue

        for validation_stat in validation_stats_to_track:
            try:
                stat_validation_events = recorded_events_accumulator.Scalars("%s/%s" % (validation_stat, valitation_set))
                stat_validation_data = [event_data.value for event_data in stat_validation_events]
                best_validation_epoch_stat_value = stat_validation_data[best_epoch_index]
                run_dict[get_stat_column_name(validation_stat, valitation_set)] = best_validation_epoch_stat_value
            except KeyError as e:
                run_dict[get_stat_column_name(validation_stat, valitation_set)] = None

        # Get hparam configuration

        hparams_config_file = join(runwise_lightning_log_subfolder, "hparams.yaml")
        if not isfile(hparams_config_file):
            warnings.warn("hparams.yaml file not found. Skipping run.")
            continue

        model_dict, datamodule_dict = get_model_and_datamodule_dicts_from_hparams_file(hparams_config_file,
                                                                                       is_full_path=True)
        datamodule_dict = modify_datamodule_config_dict(datamodule_dict, batch_size=override_batch_size_with_value,
                                                        cross_validation_fold_index=None)

        # Run tests on model

        model = None
        datamodule = None
        trainer = None
        if rerun_on_validation_set or run_on_test_set:
            checkpoint_folder = join(runwise_lightning_log_subfolder, "checkpoints")
            if not isdir(checkpoint_folder):
                warnings.warn("checkpoints folder not found. Skipping run.")
                continue

            model_checkpoint_files = [join(checkpoint_folder, checkpoint_file)
                                      for checkpoint_file in listdir(checkpoint_folder)]
            assert len(model_checkpoint_files) == 1
            model_checkpoint_file = model_checkpoint_files[0]

            trainer = get_trainer_from_config_dict(trainer_config_dict)
            model = get_model_from_checkpoint_file(model_dict, model_checkpoint_file)
            datamodule = get_datamodule_from_config_dict(datamodule_dict)

            if save_confusion_matrices:
                model.save_computed_confusion_matrices(confusion_matrices_folder, run_id)

            if rerun_on_validation_set:
                validation_results = trainer.validate(model, datamodule=datamodule)[0]
                for validation_stat in validation_stats_to_track:
                    full_stat_name = "%s/%s" % (validation_stat, valitation_set)
                    if full_stat_name in validation_results.keys():
                        run_dict[get_stat_column_name(validation_stat, valitation_set, True)]\
                            = validation_results[full_stat_name]
                    else:
                        run_dict[get_stat_column_name(validation_stat, valitation_set, True)] = None

            if run_on_test_set:
                test_results = trainer.test(model, datamodule=datamodule)[0]
                for test_stat in test_stats_to_track:
                    full_stat_name = "%s/%s" % (test_stat, test_set)
                    if full_stat_name in test_results.keys():
                        run_dict[get_stat_column_name(test_stat, test_set, True)] \
                            = test_results[full_stat_name]
                    else:
                        run_dict[get_stat_column_name(test_stat, test_set, True)] = None
                        
        # Get wanted hparams

        hparams_buffer = {}
        
        model_hparams_dict = model_dict["init_args"]
        for hparam in model_hparams_to_track:
            hparams_buffer[hparam] = get_hparam_from_dict(hparam, model_hparams_dict)

        datamodule_hparams_dict = datamodule_dict["init_args"]
        for hparam in datamodule_hparams_to_track:
            hparams_buffer[hparam] = get_hparam_from_dict(hparam, datamodule_hparams_dict)

        # To control the order of displayed hparams
        for hparam in hparams_order:
            run_dict[hparam] = hparams_buffer[hparam]

        # Additional operations

        if list_of_additional_operations is not None:
            for operation_dict in list_of_additional_operations:
                operation_name = operation_dict["name"]
                operation_kwargs = operation_dict["kwargs"]

                assert operation_name in supported_additional_operations_list

                if operation_name == "PLACEHOLDER":
                    # If you want to define an extra operation during analysis, it should go here,
                    pass

                else:
                    raise ValueError("Operation %s unsupported." % operation_name)

        # Write results onto CSV file

        with open(csv_output_filename, "a", newline="") as csv_output_file:
            field_names = list(run_dict.keys())
            writer = csv.DictWriter(csv_output_file, fieldnames=field_names)
            if write_first_line_flag:
                writer.writeheader()
                write_first_line_flag = False
            writer.writerow(run_dict)


def get_hparam_from_dict(hparam_path: str, hparams_dict: Dict[str, Any]):
    subdict_or_value = hparams_dict
    hparam_path_split = hparam_path.split("/")

    for key in hparam_path_split:
        if key not in subdict_or_value.keys():
            return None
        subdict_or_value = subdict_or_value[key]
    last_value = subdict_or_value

    if hparam_path_split[-1] == "class_path" and isinstance(last_value, str):
        last_value = last_value.split(".")[-1]

    return last_value


def get_run_results_from_file(run_analysis_config_name: str):
    if not run_analysis_config_name[-5:] == ".yaml":
        run_analysis_config_name = run_analysis_config_name + ".yaml"

    runs_analysis_configs_folder = get_execution_configs_folder_path("results_analysis")
    run_analysis_config_file = join(runs_analysis_configs_folder, run_analysis_config_name)
    assert isfile(run_analysis_config_file)

    run_analysis_config_dict = yaml.safe_load(open(run_analysis_config_file, "r"))
    best_run_stats_to_track_dict = get_best_run_stats_to_track_dict(run_analysis_config_dict["best_run_stats_to_track_file"])

    hparams_to_track_dict = {}
    if "hparams_to_track_file" in run_analysis_config_dict.keys():
        hparams_to_track_dict = get_hparams_to_track_dict(run_analysis_config_dict["hparams_to_track_file"])

    other_inputs_as_dict = {}
    if "other_inputs" in run_analysis_config_dict.keys():
        other_inputs_as_dict = run_analysis_config_dict["other_inputs"]

    get_run_results(hparams_to_track_dict, best_run_stats_to_track_dict, **other_inputs_as_dict)