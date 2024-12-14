import pandas as pd
import math
import numpy as np
from tqdm import tqdm

def date_transfer2_for_match(date_string):
    date_split = date_string.split('brainmask_')[1].split('.mgz')[0].split('_')
    year = date_split[0]
    month = date_split[1].lstrip('0')
    day = date_split[2]
    converted_date = month + '/' + day + '/' + year
    time_num = (int(year)-1950)*12.0 + int(month)
    return time_num, converted_date


def date_transfer(date_string):
    date_set = date_string.split('-')
    year = int(date_set[0])
    month = int(date_set[1])
    time = (year-1950)*12 + month  # treat month as unit
    return time


def lists_sort(state_list, visit_list, date_list, ori_date_list):
    indices = np.argsort(date_list)
    date_list_new = sorted(date_list)
    state_list_new = [state_list[index] for index in indices]
    visit_list_new = [visit_list[index] for index in indices]
    ori_date_list_new = [ori_date_list[index] for index in indices]

    return state_list_new, visit_list_new, date_list_new, ori_date_list_new


def decide_state(dataset_type, diagnosis_list, subject_id):
    '''
    detaset_type: ADNI1, ADNI2, ADNIGO, or ADNI3
    diagnosis_list: [DXCURREN, DXCHANGE, DIAGNOSIS]
    return: 'CN', 'MCI', 'AD'
    '''
    state_map1 = {1:'CN',2:'MCI',3:'AD'}
    state_map2 = {1:'CN',2:'MCI',3:'AD',
                  4:'MCI',5:'AD',6:'AD',
                  7:'CN',8:'MCI',9:'CN'}
    
    if dataset_type not in ['ADNI1','ADNI2','ADNIGO','ADNI3']:
        raise TypeError("Please input correct dataset type.")
    
    if dataset_type == 'ADNI1' and not math.isnan(diagnosis_list[0]):
        return state_map1[diagnosis_list[0]]
    elif dataset_type == 'ADNI2' or dataset_type == 'ADNIGO' and not math.isnan(diagnosis_list[1]):
        return state_map2[diagnosis_list[1]]
    elif dataset_type == 'ADNI3' and not math.isnan(diagnosis_list[2]):
        return state_map1[diagnosis_list[2]]
    else:
        return "Empty_state"
        

def patient_classify(state_list):
    if len(state_list) == state_list.count('CN'):
        patient_type = 'all_CN'
    elif len(state_list) == state_list.count('MCI'):
        patient_type = 'all_MCI'
    elif len(state_list) == state_list.count('AD'):
        patient_type = 'all_AD'
    elif len(state_list) == state_list.count('Patient'):
        patient_type = 'all_Patient'
    elif state_list.count('MCI')>0 and state_list.count('AD')>0:
        patient_type = 'pMCI'
    elif state_list.count('MCI')>0 and state_list.count('CN')>0 and state_list.count('MCI')+state_list.count('CN')==len(state_list):
        patient_type = 'CN2MCI'
    elif state_list.count('CN')>0 and state_list.count('AD')>0 and state_list.count('AD')+state_list.count('CN')==len(state_list):
        patient_type = 'CN2AD'
    else:
        patient_type = 'unknow_type'
    
    return patient_type

# take diag's visit_single as pivot.
def search_result_combine_longitudinal(data, diag_patient_dict):
    search_subject_dict = {}
    print("Starting combining searching csv data...")
    for i in tqdm(range(data.shape[0])):
        subject_id = data[i,1]
        visit_single = data[i,5]
        date_single = data[i,9]
        describe_single = data[i,7]
        # if 'Longitudinal' in describe_single: FreeSurfer Cross-Sectional Processing brainmask
        if "FreeSurfer Cross-Sectional Processing brainmask" == describe_single and subject_id in diag_patient_dict.keys():
            if subject_id not in search_subject_dict.keys():
                search_subject_dict[subject_id] = {'visit':[], 'date':[], 'state_dict':{}, 'visit_date_dict':{}, 'date_visit_dict':{}, 'descri_type':{}}  
            # if visit_single not in search_subject_dict[subject_id]['visit']:
            visit_single, state_single = find_state_by_diag_csv(visit_single, subject_id, diag_patient_dict[subject_id])
            if date_single not in search_subject_dict[subject_id]['date']:
                search_subject_dict[subject_id]['visit'].append(visit_single)
                search_subject_dict[subject_id]['date'].append(date_single)
                # if subject_id == '014_S_0520' and visit_single == 'm36':
                #     test = 1
                search_subject_dict[subject_id]['state_dict'][date_single] = state_single
                search_subject_dict[subject_id]['visit_date_dict'][date_single] = visit_single
                search_subject_dict[subject_id]['date_visit_dict'][visit_single] = date_single
                search_subject_dict[subject_id]['descri_type'][visit_single] = []
            search_subject_dict[subject_id]['descri_type'][visit_single].append(describe_single)

    return search_subject_dict


def search_result_combine_clinical(data, longitudinal_dict, diag_patient_dict):
    search_subject_dict = {}
    print("Starting combining clinical csv data...")
    for i in tqdm(range(data.shape[0])):
        subject_id = data[i,3]
        visit_single = data[i,5]
        date_single = data[i,6]
        if subject_id in longitudinal_dict.keys():
            if subject_id not in search_subject_dict.keys():
                search_subject_dict[subject_id] = {'visit':[], 'date':[], 'state_dict':{}, 'visit_date_dict':{}, 'date_visit_dict':{}, 'descri_type':{}}
            visit_single, state_single = find_state_by_diag_csv(visit_single, subject_id, diag_patient_dict[subject_id])
            if state_single == -1:
                continue
            if visit_single in longitudinal_dict[subject_id]['visit']:
                if visit_single not in search_subject_dict[subject_id]['visit']:
                    date_single = longitudinal_dict[subject_id]['date_visit_dict'][visit_single]
                    search_subject_dict[subject_id]['visit'].append(visit_single)
                    search_subject_dict[subject_id]['date'].append(date_single)
                    search_subject_dict[subject_id]['state_dict'][date_single] = state_single
                    search_subject_dict[subject_id]['visit_date_dict'][date_single] = visit_single
                    search_subject_dict[subject_id]['date_visit_dict'][visit_single] = date_single
                    search_subject_dict[subject_id]['descri_type'][visit_single] = longitudinal_dict[subject_id]['descri_type'][visit_single]
    
    return search_subject_dict

def fetch_cols_of_interest_clinical(clinical_df, multimodal_dict, diag_patient_dict):
        clinical_dict = {}
        # filter unused rows
        clinical_df = clinical_df[clinical_df['PTID'].isin(multimodal_dict.keys())]
        cols_of_interests = ['PTID', 'VISCODE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY',
                             'APOE4', 'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 
                            'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR', 'FAQ',
                            'Ventricles', 'Hippocampus', 'WholeBrain', 
                            'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']
        clinical_df = clinical_df[cols_of_interests]
        with tqdm(total=clinical_df.shape[0]) as pbar:
            pbar.set_description('converting clinical data viscode...')
            for index, row in clinical_df.iterrows():
                subject_id = row['PTID']
                visit_single = row['VISCODE']
                if subject_id not in clinical_dict.keys():
                    clinical_dict[subject_id] = {}
                visit_single, _ = find_state_by_diag_csv(visit_single, subject_id, diag_patient_dict[subject_id])
                if visit_single in multimodal_dict[subject_id]['visit']:
                    if visit_single not in clinical_dict[subject_id].keys():
                        clinical_dict[subject_id][visit_single] = row.drop(labels=['PTID', 'VISCODE']).to_frame().T

                pbar.update(1)

        return clinical_dict

def subject_match(search_subject_dict, query_subject_list):
    matched_subject = {}
    for subject in search_subject_dict.keys():
        # if subject in query_subject_list:
        if subject in query_subject_list and len(search_subject_dict[subject]['visit']) > 1:
            matched_subject[subject] = search_subject_dict[subject]
    return matched_subject

def find_state_by_diag_csv(visit_single, subject_id, single_subject_dict):
    if visit_single == 'sc':
        visit_single = 'bl'
    if visit_single == 'nv':
        visit_single = 'm36'
    if visit_single not in single_subject_dict['visit']:
        if single_subject_dict['state'][-1] == 'AD':
            state_single = 'AD'
        else:
            state_single = -1 # clinical data doesnt have matching image data, visit like m204, etc.
    else:
        index = single_subject_dict['visit'].index(visit_single)
        state_single = single_subject_dict['state'][index]
    return visit_single, state_single
