#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil
import warnings

import mne
import pyedflib
import pickle
import numpy as np
import os
import yaml
from numpy.testing import assert_almost_equal

# Legacy code, adapted from Paul Dequidt (hence the mixed English-French comments...)
# paul.dequidt@unicaen.fr - ORCID: 0000-0002-8362-7735

##################################################################################
# Ouvre les .edf de MASS, et sauve le signal utile de chaque sujet au format .pkl
# Entrée : un dossier (MASS) avec les .edf des sujets
# Sortie : les .pkl de chaque sujet dans un dossier "MASS_extracted" (ex: 0001.pkl, 0002.pkl...)
#          (le pkl contient toutes les électrodes, l'hypnogramme et la Fs)
##################################################################################


def get_list_suj(rootdir):
    """Récupère une liste avec tous les sujets MASS présents dans un dossier
    Entrée : un mass_root_dir avec les data .edf (ex:'SC4001E0-PSG.edf')
    Sortie : la liste des sujets nettoyés et non nettoyés (ex:'0001' et '01-03-0001')"""
    list_f = os.listdir(rootdir)
    list_suj_sale = []
    list_suj_propre = []
    for f in list_f:
        if '.edf' in f:  # je ne traite que les .edf (au cas où)
            suj_sale = f.split(' ')[0]
            if suj_sale not in list_suj_sale:
                list_suj_sale.append(suj_sale)
                list_suj_propre.append(suj_sale.split('-')[-1])
    return list_suj_propre, list_suj_sale


def parse_list_el(list_el, el_a_sauver):
    """Nettoie le nom des électrodes à l'aide d'une liste de référence (variable globale)
    Entrée : la liste sale
    Sortie : la liste nettoyée"""
    
    list_el_propre = []
    
    for el in list_el:
        for i in el_a_sauver:
            if i in el:
                list_el_propre.append(i)
    return list_el_propre


def suppr_signal_inutile(VT, signal_score, epoch_duration, fe, mass_groundtruth_to_our_groundtruth):
    """Supprime les epochs '?' du signal
    Entrée : La VT, le signal scoré (donc hors offset de début et fin d'epoch), la frequence d'ech
    Sortie : le signal scoré utile et sa VT (np.arrays)"""

    list_of_acceptable_labels = mass_groundtruth_to_our_groundtruth.keys()

    # compte les epochs par labels de VT
    list_i = []  # indice des epochs '?' dans le signal scoré
    for i in range(VT.shape[0]):
        assert VT[i][:-1] == "Sleep stage "
        stage_label = VT[i][-1]
        if stage_label not in list_of_acceptable_labels:
            if stage_label != "?":
                warnings.warn("Unexpected epoch label: \"%s\". Skipping epoch as if its label was \"Sleep stage ?\"." % VT[i])
            list_i.append(i)
        VT[i] = stage_label

    # Je vais parcourir le signal, en concaténant epoch par epoch, et si j'ai un '?', je passe
    signal_score_utile = np.empty((signal_score.shape[0],0))
    VT_utile=[]
    for i in range(VT.shape[0]):  # parcourt tous les epochs
        if i not in list_i:  # l'epoch n'est pas un '?'
            # Je recupère l'epoch et je concatène
            epoch = signal_score[:, int(i*epoch_duration*fe):int((i+1)*epoch_duration*fe)]
            signal_score_utile = np.concatenate((signal_score_utile,epoch), axis=1)

            # je gère la VT
            stage_label = VT[i]
            assert stage_label in list_of_acceptable_labels
            VT_utile.append(mass_groundtruth_to_our_groundtruth[stage_label])

    VT_utile = np.asarray(VT_utile, dtype=int)  # caste la liste VT en array

    nb_of_skipped_epochs = len(list_i)

    return signal_score_utile, VT_utile, nb_of_skipped_epochs


def openEDF(adressePSG, adresseVT, epoch_duration, reject_el, el_a_sauver, mass_groundtruth_to_our_groundtruth):
    
    """Nettoie le nom des électrodes à l'aide d'une liste de référence (variable globale)
    Entrée : l'adresse du fichier EDF++ du PSG et de la VT (fichier 'base')
    Sortie : Un dico sujet avec :
    -les électrodes (signal scoré brut)
    -les canaux calculés depuis la référence 'A2'
    -la vérité terrain, dicosuj['hypno']"""
    
    ##Open PSG, load les electrodes voulues et les variables utiles
    PSG = mne.io.read_raw_edf(adressePSG, exclude=reject_el)
    el_data = PSG.get_data()  # return un array (n_el,n_ech_acq)
    Nb_ech_total = el_data.shape[1]  # nb ech de tout l'enregistrement (à crop)
    fe = PSG.info['sfreq']  # fréquence d'échantillonnage

    # Récupère les électrodes disponibles et nettoie les noms
    list_el = PSG.ch_names
    list_el = parse_list_el(list_el, el_a_sauver)
    assert len(list_el) == len(el_a_sauver)
    
    ##Récupère le signal utile
    base = pyedflib.EdfReader(adresseVT)
    ann_base = base.readAnnotations()
    assert len(ann_base) == 3
    assert len(ann_base[0]) == len(ann_base[1]) == len(ann_base[2])
    Nb_epochs = len(ann_base[0])

    t_offset = ann_base[0][0] #l'offset temporel (en secondes)
    N_ech_offset = t_offset*fe
    Nb_ech_utiles = Nb_epochs*epoch_duration*fe

    # Knowing that there is a delay between the beginning of the recording and the beginning of the first epoch,
    # checking if any other such gaps exist (gaps greater than a millisecond are tolerated, because of floating-point
    # inaccuracies)
    epoch_starting_time = t_offset
    for following_epoch_index in range(1, Nb_epochs):
        following_epoch_starting_time = ann_base[0][following_epoch_index]
        epoch_starting_time += 30
        assert_almost_equal(following_epoch_starting_time, epoch_starting_time, decimal=3)

    # signal_scoré (scoré, divisible par 30 secondes)
    signal_score=el_data[:,int(N_ech_offset):int(N_ech_offset+Nb_ech_utiles)]

    ##Récupère la vérité terrain
    VT = ann_base[2]

    assert len(VT) == Nb_epochs

    ##Retire les epochs '?' de la VT, obtention du signal_scoré_utile
    signal_score_utile, VT, nb_of_skipped_epochs = suppr_signal_inutile(VT, signal_score, epoch_duration, fe, mass_groundtruth_to_our_groundtruth)
    final_nb_of_epochs = len(VT)

    assert Nb_epochs == final_nb_of_epochs + nb_of_skipped_epochs

    dicosuj={}
    dicosuj['hypno'] = VT  # Vérité Terrain
    for i in range(len(list_el)):
        dicosuj[list_el[i]] = signal_score_utile[i,:]  # ajoute les electrodes (signal scoré brute)
        assert len(dicosuj[list_el[i]]) == final_nb_of_epochs * epoch_duration * fe

    return dicosuj, fe

def spectrogramme(el,fe):
    from scipy import signal
    import matplotlib.pyplot as plt
    Dt=int(5*fe) #le temps en secondes regardé par fenêtre (ici 30sec)
    noverlap=int(Dt/2)
    f, t, Sxx = signal.spectrogram(el, fe,window=('hamming'),
                                    nperseg=Dt, 
                                    noverlap=noverlap,
                                    detrend=False)
    t=t-t[0] #je recentre t en 0
    Sxx=10*np.log10(Sxx)

    plt.figure(1,figsize=(10, 5),frameon=False)
    plt.pcolormesh(f, t, Sxx,cmap='jet',vmin=-10,vmax=20)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def show_eeg(el):
    import matplotlib.pyplot as plt
    axe_x=np.arange(0,el.shape[0],1)
    plt.figure(1,figsize=(10, 5),frameon=False)
    plt.plot(axe_x,el)
    plt.ylabel('Amplitude')
    plt.xlabel('nb_ech')
    plt.show()


def main(mass_subset_name="SS3"):
    current_script_directory = os.path.dirname(os.path.realpath(__file__))
    data_extraction_dir = os.path.dirname(current_script_directory)
    data_preprocessing_dir = os.path.dirname(data_extraction_dir)
    datasets_dir = os.path.join(data_preprocessing_dir, "_2_1_original_datasets")
    mass_data_dir = os.path.join(datasets_dir, "MASS_%s" % mass_subset_name)
    mass_save_dir = os.path.join(data_extraction_dir, "MASS_%s_extracted" % mass_subset_name)
    configs_dir = os.path.join(current_script_directory, "configs")
    config_file = os.path.join(configs_dir, "MASS_%s.yaml" % mass_subset_name)

    assert os.path.isdir(mass_data_dir)
    assert os.path.isfile(config_file)

    if os.path.exists(mass_save_dir):
        shutil.rmtree(mass_save_dir)
    os.makedirs(mass_save_dir)

    config_dict = yaml.safe_load(open(config_file, "r"))
    epoch_duration = config_dict["epoch_duration_in_seconds"]
    reject_el = config_dict["electrodes_to_reject"]
    el_a_sauver = config_dict["electrodes_to_conserve_include"]
    mass_groundtruth_to_our_groundtruth = config_dict["dataset_groundtruth_to_our_groundtruth"]

    # Je récupère tous les sujets du dossier
    list_suj_propre, list_suj = get_list_suj(mass_data_dir)
    list_suj_propre.sort()
    list_suj.sort()

    dicosuj_keys = None
    for i in range(len(list_suj)):
        # ouvre un suj, load son PSG, sa VT, divise tout en epochs, return un dico (?)
        adressePSG = os.path.join(mass_data_dir, list_suj[i] + ' PSG.edf')
        adresseVT = os.path.join(mass_data_dir, list_suj[i] + ' Base.edf')
        dicosuj, sampling_frequency = openEDF(adressePSG, adresseVT, epoch_duration, reject_el, el_a_sauver, mass_groundtruth_to_our_groundtruth)

        allowed_keys = el_a_sauver + ["hypno"]
        print(dicosuj.keys())
        dicosuj = {key: dicosuj[key] for key in allowed_keys}
        print("Keys for", list_suj_propre[i])

        dicosuj["Fs"] = sampling_frequency
        dicosuj["EEG Signals"] = el_a_sauver

        tmp_dicosuj_keys = tuple(dicosuj.keys())
        if dicosuj_keys is None:
            dicosuj_keys = tmp_dicosuj_keys
        assert tmp_dicosuj_keys == dicosuj_keys

        save_file = os.path.join(mass_save_dir, list_suj_propre[i] + '.pkl')
        pickle.dump(dicosuj, open(save_file, 'wb'))
        print(list_suj_propre[i], 'ok')

    keys_save_file = os.path.join(mass_save_dir, ".saved_keys.txt")
    with open(keys_save_file, "w") as f:
        f.write(", ".join(dicosuj_keys))


if __name__ == '__main__':
    main("SS3")


