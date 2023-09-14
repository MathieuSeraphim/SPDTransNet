from jsonargparse import ArgumentParser
from _5_execution._5_1_runs_analysis.get_run_results import get_run_results_from_file

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_analysis_config_name", type=str, default="_SPD_from_EEG_base_runs_analysis_config.yaml")
    command_line_inputs = parser.parse_args()

    get_run_results_from_file(command_line_inputs.run_analysis_config_name)



