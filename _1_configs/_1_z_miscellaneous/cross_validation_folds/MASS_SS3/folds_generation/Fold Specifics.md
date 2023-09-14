<h1 style="text-align: center;">Fold Specifics</h1>

The cross-validation `.yaml` files for the MASS-SS3 dataset have been generated from the `idx_MASS.npy`
[file](./idx_MASS.npy) by running the `IITNet_folds_to_yaml_script.py`
[Python script](./IITNet_folds_to_yaml_script.py).

The `idx_MASS.npy` file was recovered from [the GitHub repository for the IITNet model](https://github.com/gist-ailab/IITNet-official/tree/main/split_idx)
by [Hogeon Seo](https://orcid.org/0000-0002-7655-6203), [Seunghyeok Back](https://orcid.org/0000-0003-4334-9053),  [Seongju Lee](https://scholar.google.com/citations?user=Q0LR04AAAAAJ&hl=en) et al.  
It is composed of 31 seemingly randomly divided training / validation / test set splits of the MASS-SS3 dataset's 62
subjects, with respectively 50, 10 and 2 subjects per set, and no overlap between test sets.