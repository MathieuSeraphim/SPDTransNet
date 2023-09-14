import os
import pickle
import shutil


def main():

    dataset = "test"
    data_extraction_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_preprocessing_dir = os.path.dirname(data_extraction_dir)
    datasets_dir = os.path.join(data_preprocessing_dir, "_2_1_original_datasets")
    test_data_dir = os.path.join(datasets_dir, dataset)
    test_save_dir = os.path.join(data_extraction_dir, dataset + "_extracted")

    assert os.path.isdir(test_data_dir)
    assert os.listdir(test_data_dir) == ["dummy.edf",]

    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
    os.makedirs(test_save_dir)

    dict = {"None object": None}
    dict_keys = list(dict.keys())
    save_file = os.path.join(test_save_dir, "dummy.pkl")
    pickle.dump(dict, open(save_file, 'wb'))

    keys_save_file = os.path.join(test_save_dir, ".saved_keys.txt")
    with open(keys_save_file, "w") as f:
        f.write(", ".join(dict_keys))


if __name__ == '__main__':
    main()
