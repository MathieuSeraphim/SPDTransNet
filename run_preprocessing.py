import sys
from _2_data_preprocessing._2_3_preprocessors.SPD_matrices_from_EEG_signals.SPDFromEEGPreprocessor import main


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("Not multiprocessing.")
        main(sys.argv[1])
    else:
        raise ValueError("Wrong number of arguments.")






