from typing import Optional
import SimpleITK as sitk
import torchio as tio
from torch.utils.data import Dataset
import numpy as np
import torch
import SimpleITK as sitk
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager
import random
import os
import sys
from torchvision import transforms as T
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import glob
from PIL import Image
from tqdm import tqdm
import cv2
import platform
import itertools as it
import random
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import pandas as pd
import math
from datamodule.datamodule_utils import *
from pysnptools.snpreader import Bed
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.ndimage import zoom
import json


class ADNI_DataSet(Dataset):

    def __init__(self, cfg, transform, mode):
        super(ADNI_DataSet, self).__init__()
        self.cfg = cfg
        self.data_root_path = cfg.data_root_path
        self.diag_csv_path = cfg.diag_csv_path # DXSUM_PDXCONV_ADNIALL_24Sep2023
        self.data_collection_csv_path = cfg.data_collection_csv_path # ANDI1GO_postprocess_10_09_2023
        self.clinical_csv_path = cfg.clinical_path
        self.gene_path = cfg.gene_dir_path
        self.apoe_gene_path = cfg.apoe_gene_dir_path
        self.pred_xYear = cfg.pred_xMonth
        self.seq_len = cfg.seq_len
        self.TEM_a = cfg.TEM_a
        self.TEM_b = cfg.TEM_b
        self.seq_aug_flag = cfg.seq_aug_flag
        self.pad_mode = cfg.pad_mode
        self.time_emd_type = cfg.time_emd_type
        self.data_split_ratio = cfg.data_split_ratio
        self.seed = cfg.seed
        self.nor_img_flag = cfg.nor_img_flag
        self.mode = mode
        self.transform = transform
        self.is_shuffle = cfg.is_shuffle
        self.data_pct = cfg.data_pct
        self.apoe_set = {"rs4575098", "rs6656401", "rs2093760", "rs4844610", "rs4663105", "rs6733839", "rs10933431", "rs35349669",
        "rs6448453", "rs190982", "rs9271058", "rs9473117", "rs9381563", "rs10948363", "rs2718058", "rs4723711",
        "rs1859788", "rs1476679", "rs12539172", "rs10808026", "rs7810606", "rs11771145", "rs28834970", "rs73223431",
        "rs4236673", "rs9331896", "rs11257238", "rs7920721", "rs3740688", "rs10838725", "rs983392", "rs7933202",
        "rs2081545", "rs867611", "rs10792832", "rs3851179", "rs17125924", "rs17125944", "rs10498633", "rs12881735",
        "rs12590654", "rs442495", "rs59735493", "rs113260531", "rs28394864", "rs111278892", "rs3752246", "rs4147929",
        "rs41289512", "rs3865444", "rs6024870", "rs6014724", "rs7274581", "rs429358"}

        random.seed(self.seed)

        self.diag_patient_dict = self._handle_diagnosis_csv()
        if self.data_pct != 1:
            total_items = len(self.diag_patient_dict)
            num_to_select = int(total_items * self.data_pct)
            selected_keys = random.sample(list(self.diag_patient_dict.keys()), num_to_select)
            self.diag_patient_dict = {key: self.diag_patient_dict[key] for key in selected_keys}
        self.longitudinal_subject_dict = self._handle_collection_csv(self.diag_patient_dict)
        self.longitudinal_clinic_subject_dict = self._handle_clinical_csv(self.longitudinal_subject_dict, self.diag_patient_dict)
        self.longitudinal_multimodal_subject_dict = self._handle_gene(self.longitudinal_clinic_subject_dict) # merge subject
        self.get_statics(self.longitudinal_multimodal_subject_dict.keys())
        self._find_longitudinal_data() # load actual clinical data
        self._divide_subject_follow_split()
        if mode == 'train':
            self.target_subject_list = self.train_subject_list
        elif mode == 'val':
            self.target_subject_list = self.val_subject_list
        elif mode == 'test':
            self.target_subject_list = self.test_subject_list
        elif mode == 'all':
            self.target_subject_list = list(self.longitudinal_multimodal_subject_dict.keys())
        else:
            raise TypeError("Input WRONG mode!")

        self.diag_patient_dict = None
        self.longitudinal_subject_dict = None
        self.longitudinal_clinic_subject_dict = None
        self.longitudinal_multimodal_subject_dict = None
        
        self.imgpath_seq_pool = []
        self.label_seq_pool = []
        self.next_seq_label_pool = []
        self.next_label_pool = []
        self.next_cli_pool = []
        self.time_seq_pool = []
        self.next_time_pool = []
        self.pMCI_flag_pool = []
        self.imgname_seq_pool = []
        self.img_seq_pool = []
        self.gen_iids_pool = []
        self.cli_seq_pool = [] # actual clinical data
        self.constant_cli_pool = []
        self._read_seq_img_label_into_memory(self.target_subject_list,
                                            seq_len=self.seq_len,
                                            seq_aug_flag=self.seq_aug_flag,
                                            max_aug_ratio=4,
                                            pad_mode=self.pad_mode
                                            )
        
        # self.pre_load_img()


    def __getitem__(self, index):
        time_seq_single = self.time_seq_pool[index]
        next_time_single = self.next_time_pool[index]
        next_label_single = self.next_label_pool[index]
        label_seq_single = self.label_seq_pool[index]
        next_seq_label_single = self.next_seq_label_pool[index]
        next_cli_single = self.next_cli_pool[index]
        constant_cli_single = self.constant_cli_pool[index]
        imgpath_seq_single = self.imgpath_seq_pool[index]['imgpath']
        imgname_seq_single = self.imgname_seq_pool[index]
        gen_seq_single = self.gen_iids_pool[index]
        cli_seq_single = self.cli_seq_pool[index]

        img_seq_single = []
        for imgpath in imgpath_seq_single:
            if imgpath == 'all_zeros_img':
                img_seq_single.append(np.zeros(128,128,128), dtype=np.float32)
            else:
                img = nib.load(imgpath)
                data = img.get_fdata()
                if self.nor_img_flag:
                    data = data / 255.0
                zoom_factors = (128.0/256, 128.0/256, 128.0/256)
                downsampled_data = zoom(data, zoom_factors, mode='nearest')
                img_seq_single.append(downsampled_data.astype(np.float32))
        img_seq_single = np.array(img_seq_single)

        time_seq_single = np.append(time_seq_single, next_time_single).astype(np.float32)
        time_seq_single_Nor = time_seq_single / (40.0) # the maximum date month is 36
        time_matrix_single_1 = np.zeros((self.seq_len, self.seq_len))
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                time_matrix_single_1[i][j] = time_seq_single_Nor[i] - time_seq_single_Nor[j]
                
        time_matrix_single = 1.0 / (1+np.exp(self.TEM_a*time_matrix_single_1-self.TEM_b))

        sample_seq = {'image':img_seq_single, 
                      'imgpath_seq':imgpath_seq_single,
                      'delta_year':time_seq_single_Nor, 
                      'next_clinical': next_cli_single,
                      'next_label':next_label_single, 
                      'label_seq':label_seq_single,
                      'next_seq_label':next_seq_label_single,
                      'Tmatrix':time_matrix_single,
                      'img_name':imgname_seq_single,
                      'gene':gen_seq_single,
                      'clinical':cli_seq_single,
                      'constant_cli_single': constant_cli_single}

        return sample_seq
    
    


    def __len__(self):
        return len(self.time_seq_pool)
    

    def get_label_pool(self):
        return self.next_seq_label_pool


    def pre_load_img(self):
        with tqdm(total=len(self.imgpath_seq_pool)) as pbar:
            pbar.set_description('Pre loading img')
            for imgpath_seq_dict in self.imgpath_seq_pool[:4]:
                img_seq = []
                imgpath_seq = imgpath_seq_dict['imgpath']
                for imgpath in imgpath_seq:
                    if imgpath == 'all_zeros_img':
                        img_seq.append(np.zeros(128,128,128))
                    else:
                        img = nib.load(imgpath)
                        data = img.get_fdata()
                        if self.nor_img_flag:
                            data = data / 255.0
                        img_seq.append(data)
                self.img_seq_pool.append(np.array(img_seq))
                pbar.update(1)        


    def _read_seq_img_label_into_memory(self, 
                                        target_subject_list,
                                        seq_len, 
                                        seq_aug_flag, 
                                        max_aug_ratio=20, 
                                        pad_mode="ZEROS"):
        if self.mode == 'train' or self.mode == 'all':
            # Concatenate all the subject clinical data into a single DataFrame
            clinical_df = None
            for subject_id in target_subject_list:
                clinical_seq = pd.concat(self.subject_clinical_dict[subject_id], ignore_index=True)  # a list of dataframe
                if clinical_df is None:
                    clinical_df = clinical_seq
                else:
                    clinical_df = pd.concat([clinical_df, clinical_seq], ignore_index=True)
            
            # Fit the preprocessor with the DataFrame
            self.transform(True, clinical_df)

        # Now transform each clinical sequence using the fitted preprocessor
        for subject_id in target_subject_list:
            clinical_seq_df = pd.concat(self.subject_clinical_dict[subject_id], ignore_index=True)
            normalized_clinical_seq = self.transform(False, clinical_seq_df)
            normalized_clinical_seq_list = normalized_clinical_seq.tolist()
            self.subject_clinical_dict[subject_id] = normalized_clinical_seq_list

        with tqdm(total=len(target_subject_list)) as pbar:
            pbar.set_description('Reading Seq into memory')
            for subject_id in target_subject_list:
                imgpath_seq = self.subject_imgpath_dict[subject_id]
                imgname_seq = self.subject_imgname_dict[subject_id]
                label_seq = self.subject_label_dict[subject_id]
                time_seq = self.subject_time_dict[subject_id]
                pMCI_flag = self.subject_pMCI_dict[subject_id]
                clinical_seq = self.subject_clinical_dict[subject_id]
                genetic = self.subject_genetic_dict[subject_id]
                
                seq_end_id = len(label_seq) 
                
                for i in range(1,seq_end_id):
                    if i-seq_len<0:
                        if pad_mode == "ZEROS":
                            pad_imgpath = ['all_zeros_img' for i in range(seq_len-i)]
                            pad_imgname = ['PAD_imgname' for i in range(seq_len-i)]
                            pad_time = [-1 for i in range(seq_len-i)]
                            pad_seq_label = [0 for i in range(seq_len-i)]
                            pad_clinical = [[0 for _ in clinical_seq[0]] for i in range(seq_len-i)]
                        elif pad_mode == "REPLICATE":
                            pad_imgpath = [imgpath_seq[0] for i in range(seq_len-i)]
                            pad_imgname = [imgname_seq[0] for i in range(seq_len-i)]
                            pad_time = [0 for i in range(seq_len-i)]
                            pad_seq_label = [label_seq[0] for i in range(seq_len-i)]
                            pad_clinical = [clinical_seq[0] for i in range(seq_len-i)]
                        elif pad_mode == "NO_PAD":
                            pad_imgpath = "NO_PAD"
                            pad_imgname = "NO_PAD"
                            pad_time = "NO_PAD"
                            pad_seq_label = "NO_PAD"
                            pad_clinical = "NO_PAD"
                        else:
                            raise ValueError("Please choose right PADDING type!")
                    else: pad_imgpath = None
                    if pad_imgpath is not None and pad_mode != "NO_PAD":
                        self.imgpath_seq_pool.append({'imgpath':pad_imgpath+imgpath_seq[:i]})
                        self.imgname_seq_pool.append({'seq name':pad_imgname+imgname_seq[:i]})
                        self.label_seq_pool.append(np.array(pad_seq_label+label_seq[:i]))
                        self.next_seq_label_pool.append(np.array(pad_seq_label[:-1]+label_seq[:i+1]))
                        self.next_label_pool.append(label_seq[i])
                        self.next_cli_pool.append(np.array(clinical_seq[i]).astype(np.float32))
                        if len(clinical_seq) < self.seq_len:
                            self.constant_cli_pool.append(np.array(clinical_seq[0]).astype(np.float32))
                        else:
                            self.constant_cli_pool.append(np.array(clinical_seq[0]).astype(np.float32))
                        self.time_seq_pool.append(np.array(pad_time+time_seq[:i]))
                        self.next_time_pool.append(time_seq[i])
                        self.pMCI_flag_pool.append(pMCI_flag)
                        self.gen_iids_pool.append(genetic)
                        self.cli_seq_pool.append(np.array(pad_clinical+clinical_seq[:i]).astype(np.float32))
                        
                    elif pad_imgpath is not None and pad_mode == "NO_PAD":
                        continue                   
                    
                    else:
                        self.imgpath_seq_pool.append({'imgpath':imgpath_seq[i-seq_len:i]})
                        self.imgname_seq_pool.append({'seq name':imgname_seq[i-seq_len:i]})
                        self.label_seq_pool.append(np.array(label_seq[i-seq_len:i]))
                        self.next_seq_label_pool.append(np.array(label_seq[i-seq_len+1:i+1]))
                        self.next_label_pool.append(label_seq[i])
                        self.next_cli_pool.append(np.array(clinical_seq[i]).astype(np.float32))
                        self.constant_cli_pool.append(np.array(clinical_seq[self.seq_len-1]).astype(np.float32))
                        self.time_seq_pool.append(np.array(time_seq[i-seq_len:i]))
                        self.next_time_pool.append(time_seq[i])
                        self.pMCI_flag_pool.append(pMCI_flag)
                        self.gen_iids_pool.append(genetic)
                        self.cli_seq_pool.append(np.array(clinical_seq[i-seq_len:i]).astype(np.float32))
                            
                        # sequential augmentation.
                        if seq_aug_flag and pMCI_flag == True: 
                            all_idx = []
                            for e in it.combinations(np.arange(i-1),seq_len-1):
                                all_idx.append(list(e))
                            if len(all_idx)>max_aug_ratio:
                                all_idx = random.sample(all_idx,max_aug_ratio)
                            for idx in all_idx:
                                idx.append(i-1)
                                self.next_label_pool.append(label_seq[i])
                                self.next_seq_label_pool.append(np.array([label_seq[i+1] for i in idx]))
                                self.next_cli_pool.append(np.array(clinical_seq[i]).astype(np.float32))
                                self.constant_cli_pool.append(np.array(clinical_seq[self.seq_len-1]).astype(np.float32))
                                self.label_seq_pool.append(np.array([label_seq[i] for i in idx]))
                                self.imgpath_seq_pool.append({'imgpath':[imgpath_seq[i] for i in idx]})
                                self.imgname_seq_pool.append({'seq name':[imgname_seq[i] for i in idx]})
                                self.time_seq_pool.append(np.array([time_seq[i] for i in idx]))
                                self.next_time_pool.append(time_seq[i])
                                self.pMCI_flag_pool.append(pMCI_flag)
                                self.gen_iids_pool.append(genetic)
                                self.cli_seq_pool.append(np.array([clinical_seq[i] for i in idx]).astype(np.float32))
        
                pbar.update(1)       

        # shuffle the data
        if self.is_shuffle:
            indices = list(range(len(self.next_label_pool)))
            random.shuffle(indices)
            self.next_label_pool = [self.next_label_pool[i] for i in indices]
            self.next_seq_label_pool = [self.next_seq_label_pool[i] for i in indices]
            self.next_cli_pool = [self.next_cli_pool[i] for i in indices]
            self.constant_cli_pool = [self.constant_cli_pool[i] for i in indices]
            self.label_seq_pool = [self.label_seq_pool[i] for i in indices]
            self.imgpath_seq_pool = [self.imgpath_seq_pool[i] for i in indices]
            self.imgname_seq_pool = [self.imgname_seq_pool[i] for i in indices]
            self.time_seq_pool = [self.time_seq_pool[i] for i in indices]
            self.next_time_pool = [self.next_time_pool[i] for i in indices]
            self.pMCI_flag_pool = [self.pMCI_flag_pool[i] for i in indices]
            self.gen_iids_pool = [self.gen_iids_pool[i] for i in indices]
            self.cli_seq_pool = [self.cli_seq_pool[i] for i in indices]


    def _divide_subject_follow_split(self):
        all_subject_list = list(self.longitudinal_multimodal_subject_dict.keys())
        pMCI_subject_list = list(set(self.pMCI_list) & set(all_subject_list))
        other_subject_list = list(set(all_subject_list) - set(pMCI_subject_list))
        pMCI_subject_list.sort()
        other_subject_list.sort()
        split1 = int(len(pMCI_subject_list)*self.data_split_ratio[0])
        split2 = int(len(pMCI_subject_list)*(self.data_split_ratio[0]+self.data_split_ratio[1]))
        split3 = int(len(other_subject_list)*self.data_split_ratio[0])
        split4 = int(len(other_subject_list)*(self.data_split_ratio[0]+self.data_split_ratio[1]))
        self.train_subject_list = pMCI_subject_list[:split1] + other_subject_list[:split3]
        self.val_subject_list = pMCI_subject_list[split1:split2] + other_subject_list[split3:split4]
        self.test_subject_list = pMCI_subject_list[split2:] + other_subject_list[split4:]
        random.shuffle(self.train_subject_list)
        random.shuffle(self.val_subject_list)
        random.shuffle(self.test_subject_list)


    def _handle_diagnosis_csv(self, display_all_info=False):
        df = pd.read_csv(self.diag_csv_path)
        data = df.values
        patient_dict = {}
        state_tmp_list = []
        visit_tmp_list = []
        date_tmp_list = []
        self.all_CN_list = []
        self.all_MCI_list = []
        self.all_AD_list =[]
        self.pMCI_list = []
        self.CN2MCI_list = []
        self.CN2AD_list = []
        self.unknow_list = []

        print("Rearange data...")
        for i in tqdm(range(data.shape[0])):
            subject_id = data[i,3]
            dataset_type = data[i,0]
            diagnosis_info = [data[i,12],data[i,11],data[i,-2]]

            if subject_id not in patient_dict.keys():
                patient_dict[subject_id] = {'type':None,'state':[], 'visit':[], 'date':[], 'ori_date':[]}  

            subject_state = decide_state(dataset_type, diagnosis_info, subject_id)
            if subject_state != "Empty_state":
                patient_dict[subject_id]['state'].append(subject_state)
                patient_dict[subject_id]['visit'].append(data[i,5])
                if  isinstance(data[i,7], float) and math.isnan(data[i,7]):
                    patient_dict[subject_id]['date'].append(data[i,8])
                else:
                    patient_dict[subject_id]['date'].append(data[i,7])
        
        # sort each patient dict based on date
        print("Sorting patient...")
        for subject_id in tqdm(patient_dict.keys()):
            state_tmp_list = patient_dict[subject_id]['state']
            visit_tmp_list = patient_dict[subject_id]['visit']
            date_tmp_list = patient_dict[subject_id]['date']

            date_tmp_list2 = [date_transfer(date_single) for date_single in date_tmp_list]
            state_list_new, visit_list_new, date_list_new, ori_date_list_new = lists_sort(state_tmp_list, visit_tmp_list, date_tmp_list2, date_tmp_list)
            patient_type = patient_classify(state_list_new)
            patient_dict[subject_id]['state'] = state_list_new
            patient_dict[subject_id]['visit'] = visit_list_new
            patient_dict[subject_id]['date'] = date_list_new
            patient_dict[subject_id]['ori_date'] = ori_date_list_new
            patient_dict[subject_id]['type'] = patient_type

            if patient_type == 'all_CN':
                self.all_CN_list.append(subject_id)
            if patient_type == 'all_MCI':
                self.all_MCI_list.append(subject_id)
            if patient_type == 'all_AD':
                self.all_AD_list.append(subject_id)
            if patient_type == 'pMCI':
                self.pMCI_list.append(subject_id)
            if patient_type == 'CN2MCI':
                self.CN2MCI_list.append(subject_id)
            if patient_type == 'CN2AD':
                self.CN2AD_list.append(subject_id)
            if patient_type == 'unknow_type':
                self.unknow_list.append(subject_id)

        return patient_dict
    
    def get_statics(self, patient_list):
        nall_CN_list, nall_MCI_list, nall_AD_list, npMCI_list, nCN2MCI_list, nCN2AD_list, nunknow_list = [],[],[],[],[],[],[]

        for subject_id in patient_list:
            if subject_id in self.all_CN_list:
                nall_CN_list.append(subject_id)
            elif subject_id in self.all_MCI_list:
                nall_MCI_list.append(subject_id)
            elif subject_id in self.all_AD_list:
                nall_AD_list.append(subject_id)
            elif subject_id in self.pMCI_list:
                npMCI_list.append(subject_id)
            elif subject_id in self.CN2MCI_list:
                nCN2MCI_list.append(subject_id)
            elif subject_id in self.CN2AD_list:
                nCN2AD_list.append(subject_id)
            elif subject_id in self.unknow_list:
                nunknow_list.append(subject_id)

        print('There are {} patients in total.' .format(len(patient_list)))
        print('There are {} all CN patients.' .format(len(nall_CN_list)))
        print('There are {} all MCI patients.' .format(len(nall_MCI_list)))
        print('There are {} all AD patients.' .format(len(nall_AD_list)))
        print('There are {} pMCI patients.' .format(len(npMCI_list)))
        print('There are {} CN to MCI patients.' .format(len(nCN2MCI_list)))
        print('There are {} CN to AD patients.' .format(len(nCN2AD_list)))
        print('There are {} unknow patients.' .format(len(nunknow_list)))

        self.all_CN_list = nall_CN_list
        self.all_MCI_list = nall_MCI_list
        self.all_AD_list = nall_AD_list
        self.pMCI_list = npMCI_list
        self.CN2MCI_list = nCN2MCI_list
        self.CN2AD_list = nCN2AD_list
        self.unknow_list = nunknow_list
    
    def _handle_collection_csv(self, diag_patient_dict):
        df_collection_csv = pd.read_csv(self.data_collection_csv_path)
        data_collection = df_collection_csv.values
        collection_subject_dict = search_result_combine_longitudinal(data_collection, diag_patient_dict)
        matched_subject = subject_match(collection_subject_dict, self.diag_patient_dict.keys())
        return matched_subject

    def _handle_clinical_csv(self, longitudinal_dict, diag_patient_dict):
        clinical_csv = pd.read_csv(self.clinical_csv_path, low_memory=False)
        data_collection = clinical_csv.values
        collection_subject_dict = search_result_combine_clinical(data_collection, longitudinal_dict, diag_patient_dict)
        return collection_subject_dict
    
    def _handle_gene(self, clinical_dict):
        bed = Bed(self.gene_path, count_A1=False)
        collection_subject_dict = {}
        print("Starting combining genetic data...")
        for i in tqdm(range(bed.iid.shape[0])):
            subject_id = bed.iid[i,1]
            if subject_id in clinical_dict.keys():
                if subject_id not in collection_subject_dict.keys():
                    collection_subject_dict[subject_id] = clinical_dict[subject_id]
        matched_subject = subject_match(collection_subject_dict, self.diag_patient_dict.keys())
        return matched_subject
    
    def _get_all_gen_data(self):
        print('starting loading gene data...')
        all_CN_df = self._get_gen_data(self.all_CN_list)
        all_MCI_df = self._get_gen_data(self.all_MCI_list)
        all_AD_df = self._get_gen_data(self.all_AD_list)
        pMCI_df = self._get_gen_data(self.pMCI_list)
        CN2MCI_df = self._get_gen_data(self.CN2MCI_list)
        CN2AD_df = self._get_gen_data(self.CN2AD_list)

        dfs = [all_CN_df, all_MCI_df, all_AD_df, pMCI_df, CN2MCI_df, CN2AD_df]

        # Concatenate them
        genetic_df = pd.concat(dfs, ignore_index=False)

        original_columns = set(genetic_df.columns)
        min_count = int(0.4 * genetic_df.shape[0]) # drop columns that have more than 40% NaN values.
        genetic_df = genetic_df.dropna(axis=1, thresh=min_count)

        for column in genetic_df:
            if genetic_df[column].isnull().any():
                genetic_df[column] = genetic_df[column].fillna(genetic_df[column].mode()[0])

        gen_lookup = defaultdict(
            lambda: None, [(iid, idx) for idx, iid in enumerate(genetic_df.index)])

        gen = genetic_df.to_numpy()

        print('number of gene for each patient:', gen.shape[1])

        return gen, gen_lookup

    def _get_gen_data(self, ids, chromos=[i for i in range(1, 23)], sid_slice=slice(0, None, 1000)):
        gen_fillna = True

        # get APOE data from PLINK_APOE
        bed = Bed(self.apoe_gene_path, count_A1=False)
        apoe_list = list(self.apoe_set.intersection(set(bed.sid)))
        iid_to_ind_map = {iid[1][1]: iid[1][0] for iid in enumerate(bed.iid)}
        ind = bed.iid_to_index([[iid_to_ind_map.get(s2), s2] for s2 in ids if s2 in iid_to_ind_map])
        filtered_ids = [s2 for s2 in ids if s2 in iid_to_ind_map]
        sid_ind = [np.where(bed.sid == item)[0][0] for item in apoe_list]
        labels = bed[ind, sid_ind].read().val
        full_df = pd.DataFrame(index=filtered_ids, data=labels, columns=bed.sid[sid_ind], dtype=np.float32)

        # get APOE data from ADNI_1_GWAS_Plink
        bed = Bed(self.gene_path, count_A1=False)
        apoe_list = list(self.apoe_set.intersection(set(bed.sid)))
        iid_to_ind_map = {iid[1][1]: iid[1][0] for iid in enumerate(bed.iid)}
        ind = bed.iid_to_index([[iid_to_ind_map[s2], s2] for s2 in ids])
        sid_ind = [np.where(bed.sid == item)[0][0] for item in apoe_list]
        labels = bed[ind, sid_ind].read().val
        df = pd.DataFrame(index=ids, data=labels, columns=bed.sid[sid_ind], dtype=np.float32)
        full_df = full_df.combine_first(df)

        genetic_df = full_df
        if gen_fillna:
            for column in genetic_df:
                mode_value = genetic_df[column].mode()
                if not mode_value.empty:
                    if genetic_df[column].isnull().any():
                        genetic_df[column] = genetic_df[column].fillna(genetic_df[column].mode()[0])
        
        return genetic_df

        
    def _find_longitudinal_data(self, gen_fillna=True):
        state2num = {'CN':0,
                     'MCI':1,
                     'AD':2}
        self.subject_imgpath_dict = {}
        self.subject_label_dict = {}
        self.subject_time_dict = {}
        self.subject_pMCI_dict = {}
        self.subject_imgname_dict = {}
        self.subject_clinical_dict = {}
        self.subject_genetic_dict = {}
        subject_list = os.listdir(self.data_root_path)
        subject_list.sort()
        # handle genetics
        gen, gen_lookup = self._get_all_gen_data()
        
        # handle clinical data
        clinical_df = pd.read_csv(self.clinical_csv_path, low_memory=False)
        clinical_dict = fetch_cols_of_interest_clinical(
            clinical_df, self.longitudinal_multimodal_subject_dict, self.diag_patient_dict)
        
        does_not_cnt = 0

        with tqdm(total=len(subject_list)) as pbar:
            pbar.set_description('Reorganizing longitudinal data')
            for subject in subject_list:
                if subject in self.longitudinal_multimodal_subject_dict.keys():
                    self.subject_imgpath_dict[subject] = []
                    self.subject_label_dict[subject] = []
                    self.subject_time_dict[subject] = []
                    self.subject_imgname_dict[subject] = []
                    self.subject_clinical_dict[subject] = []
                    self.subject_genetic_dict[subject] = []

                    subdir_path = os.path.join(self.data_root_path,subject)
                    subject_seqtime_list = os.listdir(subdir_path)
                    subject_seqtime_list.sort()
                    init_time_num, _ = date_transfer2_for_match(subject_seqtime_list[0])
                    for file_single in subject_seqtime_list:
                        time_num, converted_date = date_transfer2_for_match(file_single)
                        # does not have clinical data for this visit.
                        if converted_date not in self.longitudinal_clinic_subject_dict[subject]['state_dict']:
                            does_not_cnt += 1
                            continue
                        if file_single.endswith('.mgz'):
                            self.subject_imgpath_dict[subject].append(os.path.join(subdir_path, file_single))
                        
                        state = self.longitudinal_multimodal_subject_dict[subject]['state_dict'][converted_date]
                        state_coverted = state2num[state]
                        visit = self.longitudinal_multimodal_subject_dict[subject]['visit_date_dict'][converted_date]
                        self.subject_label_dict[subject].append(state_coverted)
                        self.subject_time_dict[subject].append(time_num-init_time_num)
                        self.subject_imgname_dict[subject].append(subject+'-'+visit)
                        self.subject_clinical_dict[subject].append(clinical_dict[subject][visit])

                    if subject in self.pMCI_list:
                        self.subject_pMCI_dict[subject] = True
                    else:
                        self.subject_pMCI_dict[subject] = False
                    
                    gen_idx = gen_lookup[subject]
                    self.subject_genetic_dict[subject] = np.full(
                        gen.shape[1], np.nan) if gen_idx is None else gen[gen_idx]

                pbar.update(1)
        
        print('there are totally', does_not_cnt, 'images that does not have clinical data...')


    def get_dataset_stat(self):
        num_pMCI = 0
        num_all_CN = 0
        num_all_MCI = 0
        num_all_AD = 0
        num_all_CN2MCI = 0
        num_all_CN2AD = 0
        other_sub = 0
        sum_months = 0
        avg_months = 0
        num_scans = [0,0,0]  # NC, MCI, AD
        
        for subject in self.target_subject_list:
            if subject in self.pMCI_list:
                num_pMCI += 1
            elif subject in self.all_CN_list:
                num_all_CN += 1
            elif subject in self.all_MCI_list:
                num_all_MCI += 1
            elif subject in self.all_AD_list:
                num_all_AD += 1
            elif subject in self.CN2MCI_list:
                num_all_CN2MCI += 1
            elif subject in self.CN2AD_list:
                num_all_CN2AD += 1
            else:
                other_sub += 1
            for i in range(len(self.subject_label_dict[subject])):
                num_scans[self.subject_label_dict[subject][i]] += 1
                sum_months += self.subject_time_dict[subject][i]
        avg_months = float(sum_months) / float(sum(num_scans))
        clip_num_info = [0,0,0]  # NC_clip, MCI_clip, AD_clip
        for single_seq in self.next_seq_label_pool:
            for i in range(self.seq_len):
                clip_num_info[single_seq[i]] += 1

        print('There are {} patients in total.' .format(len(self.target_subject_list)))
        print('There are {} all CN patients.' .format(num_all_CN))
        print('There are {} all MCI patients.' .format(num_all_MCI))
        print('There are {} all AD patients.' .format(num_all_AD))
        print('There are {} pMCI patients.' .format(num_pMCI))
        print('There are {} CN to MCI patients.' .format(num_all_CN2MCI))
        print('There are {} CN to AD patients.' .format(num_all_CN2AD))
        print('There are {} unknow patients.' .format(other_sub))
            
        data_dict = {'pMCI':num_pMCI,'other_sub':other_sub,'avg_months':avg_months,'num_scans':num_scans, 'num_clip':clip_num_info}
        return data_dict
    