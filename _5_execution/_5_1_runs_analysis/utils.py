from os.path import join, isfile
import yaml
from _5_execution.utils import get_execution_configs_folder_path


def get_hparams_to_track_dict(hparams_to_track_file: str):
    results_analysis_configs_folder = get_execution_configs_folder_path("results_analysis")
    hparams_to_track_folder = join(results_analysis_configs_folder, "hparams_to_track")
    hparams_to_track_file = join(hparams_to_track_folder, hparams_to_track_file)
    assert isfile(hparams_to_track_file)
    return yaml.safe_load(open(hparams_to_track_file, "r"))


def get_best_run_stats_to_track_dict(best_run_stats_to_track_file: str):
    results_analysis_configs_folder = get_execution_configs_folder_path("results_analysis")
    stats_to_track_folder = join(results_analysis_configs_folder, "stats_to_track")
    best_run_stats_to_track_file = join(stats_to_track_folder, best_run_stats_to_track_file)
    assert isfile(best_run_stats_to_track_file)
    return yaml.safe_load(open(best_run_stats_to_track_file, "r"))
