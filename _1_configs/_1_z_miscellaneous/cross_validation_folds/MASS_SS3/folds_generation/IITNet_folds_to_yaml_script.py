import random
from copy import deepcopy
from os.path import dirname, realpath, join
import numpy as np
import ruamel.yaml
yaml = ruamel.yaml.YAML()


# Taken from https://stackoverflow.com/questions/63364894/how-to-dump-only-lists-with-flow-style-with-pyyaml-or-ruamel-yaml
def flist(x):
    retval = ruamel.yaml.comments.CommentedSeq(x)
    retval.fa.set_flow_style()  # fa -> format attribute
    return retval


def main(shuffle_folds: bool = False, random_seed: int = 42):

    current_script_directory = dirname(realpath(__file__))
    folds_destination_folder = dirname(current_script_directory)
    splits_filename = join(current_script_directory, "idx_MASS.npy")
    splits = np.load(splits_filename, allow_pickle=True)

    total_number_of_recordings = None
    number_of_training_set_recordings = None
    number_of_validation_set_recordings = None
    number_of_test_set_recordings = None
    list_of_all_test_set_recordings = []
    list_of_all_recording_dicts = []

    for split in splits:
        if total_number_of_recordings is None:
            number_of_training_set_recordings = len(split["train"])
            number_of_validation_set_recordings = len(split["val"])
            number_of_test_set_recordings = len(split["test"])
            total_number_of_recordings = number_of_training_set_recordings + number_of_validation_set_recordings + number_of_test_set_recordings

        # All folds have the same number of recordings
        assert len(split["train"]) == number_of_training_set_recordings
        assert len(split["val"]) == number_of_validation_set_recordings
        assert len(split["test"]) == number_of_test_set_recordings

        list_of_training_set_recordings = split["train"].tolist()
        list_of_validation_set_recordings = split["val"].tolist()
        list_of_test_set_recordings = split["test"].tolist()

        # No repetition, no overlap between test sets
        for training_set_recording in list_of_training_set_recordings:
            assert training_set_recording not in list_of_validation_set_recordings
            assert training_set_recording not in list_of_test_set_recordings
        for test_set_recording in list_of_test_set_recordings:
            assert test_set_recording not in list_of_validation_set_recordings
            assert test_set_recording not in list_of_all_test_set_recordings
        list_of_all_test_set_recordings += list_of_test_set_recordings

        # All recordings are indices (numbers in the range of total_number_of_recordings)
        for recording in list_of_training_set_recordings + list_of_validation_set_recordings + list_of_test_set_recordings:
            assert isinstance(recording, int)
            assert recording in range(total_number_of_recordings)

        fold_dict = {"training": flist(list_of_training_set_recordings),
                     "validation": flist(list_of_validation_set_recordings),
                     "test": flist(list_of_test_set_recordings)}
        list_of_all_recording_dicts.append(fold_dict)

    if shuffle_folds:
        old_list_of_all_recording_dicts = deepcopy(list_of_all_recording_dicts)
        random.seed(random_seed)
        random.shuffle(list_of_all_recording_dicts)
        assert list_of_all_recording_dicts != old_list_of_all_recording_dicts

    number_of_folds = len(list_of_all_recording_dicts)

    for fold_index in range(number_of_folds):
        fold_filename = join(folds_destination_folder, "fold_%s.yaml" % str(fold_index).zfill(2))
        with open(fold_filename, "w") as f:
            yaml.dump(list_of_all_recording_dicts[fold_index], f)


if __name__ == "__main__":
    main(shuffle_folds=False)
