import os
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
import pickle

def process_train_datasets(dataset_dir, label_file, train_output_file, val_output_file, split_file=None):
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    label_map = {}
    for item in labels:
        if 'patient_id' in item:
            label_map[item['patient_id']] = item
    
    valid_patients = set()
    for item in labels:
        if 'patient_id' in item:
            valid_patients.add(item['patient_id'])
            
    if split_file is not None:
        split_dict = torch.load(split_file)
        train_pids = set([int(pid) for pid in split_dict['train']])
        val_pids = set([int(pid) for pid in split_dict['val']])
    else:
        train_pids = valid_patients
        val_pids = set()
            
    train_records = {}
    val_records = {}
    
    patient_dirs = sorted([d for d in os.listdir(dataset_dir) if d.isdigit()])
    
    for p_dir in tqdm(patient_dirs, desc="Processing patients"):
        pid = int(p_dir)
        
        if pid not in valid_patients:
            continue
            
        if pid in train_pids:
            target_records = train_records
        elif pid in val_pids:
            target_records = val_records
        else:
            raise ValueError(f"Patient ID {pid} found in dataset but not in train/val split. Please check the split file.")
            
        p_path = os.path.join(dataset_dir, p_dir)
        p_label = label_map[pid]
        
        for vid_dir in sorted(os.listdir(p_path)):
            vid_path = os.path.join(p_path, vid_dir)
            if not os.path.isdir(vid_path):
                raise ValueError(f"Expected a directory for video sequence, but found: {vid_path}")
                
            frame_files = sorted(glob.glob(os.path.join(vid_path, 'frame_*.json')))
            if not frame_files:
                raise ValueError(f"No frame JSON files found in {vid_path}")
                
            sequence_keypoints = []
            sequence_scores = []
            sequence_obj_ids = []
            
            for f_file in frame_files:
                with open(f_file, 'r') as f:
                    try:
                        frame_data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    
                if len(frame_data['instance_info']) == 1:
                    instance = frame_data['instance_info'][0] 
                    
                    kp = np.array(instance['keypoints'][5:23])
                    assert kp.ndim == 2 and kp.shape[1] == 2, f"Expected keypoints to be a 2D array with shape (K, 2), but got {kp.shape} in {f_file}"
                    scores = np.array(instance['keypoint_scores'][5:23])
                    obj_id = instance.get('obj_id', -1)
                    # if instance.get('gt_bbox_xywh_px') is not None:
                    #     bbox = np.array(instance['gt_bbox_xywh_px'])
                    # else:
                    #     raise ValueError(f"gt_bbox_xywh_px not found in {f_file}")
                
                elif len(frame_data['instance_info']) > 1:
                    raise ValueError(f"Multiple instances found in {f_file}, expected only one. Please check the data format.")
                    
                else:
                    print(f"Warning: No instances found in {f_file}, skipping next all frames.")
                    break
                    # raise ValueError(f"No instances found in {f_file}, expected at least one. Please check the data format.")
                    
                sequence_keypoints.append(kp)
                sequence_scores.append(scores)
                sequence_obj_ids.append(obj_id)
                # sequence_bboxes.append(bbox)
                
            assert len(sequence_keypoints) == len(sequence_scores), f"Data length mismatch in {vid_path}"
            target_records[vid_dir] = {
                'keypoints': np.stack(sequence_keypoints),         # Shape: (T, K, 2)
                'keypoint_scores': np.stack(sequence_scores),      # Shape: (T, K)
                'obj_ids': np.array(sequence_obj_ids),             # Shape: (T,)
                # 'gt_bbox_xywh_px': np.stack(sequence_bboxes),
                'total_frames': len(sequence_keypoints),      
                'label': p_label
            }
            assert target_records[vid_dir]['keypoints'].shape[1] == target_records[vid_dir]['keypoint_scores'].shape[1] == 18, \
                f"Expected 18 keypoints, but got {target_records[vid_dir]['keypoints'].shape[1]} in {vid_path}"
                
    if split_file is None:
        train_output_file = train_output_file.replace('.pkl', '_all.pkl')
        print("No split file provided, all data for training")
        
    print(f"Processed {len(train_records)} training video sequences.")
    print(f"Saving to {train_output_file}...")
    with open(train_output_file, 'wb') as f:
        pickle.dump(train_records, f)
        
    if split_file is not None:
        print(f"Processed {len(val_records)} validation video sequences.")
        print(f"Saving to {val_output_file}...")
        with open(val_output_file, 'wb') as f:
            pickle.dump(val_records, f)
        
    print("Done!")


def process_test_datasets(dataset_dir, track_list):
    for pid in tqdm(track_list, desc="Processing test patients"):
        dataset_records = {}
        output_file = os.path.join(dataset_dir, f"test_track1_{pid}.pkl")
        pid_str = f"{pid:04d}" 
        p_path = os.path.join(dataset_dir, pid_str)
        
        if not os.path.exists(p_path):
            print(f"Warning: Directory {p_path} does not exist, skipping.")
            continue
        
        for vid_dir in sorted(os.listdir(p_path)):
            vid_path = os.path.join(p_path, vid_dir)
            if not os.path.isdir(vid_path):
                continue
                
            frame_files = sorted(glob.glob(os.path.join(vid_path, 'frame_*.json')))
            if not frame_files:
                continue
                
            sequence_keypoints = []
            sequence_scores = []
            sequence_obj_ids = []

            for f_file in frame_files:
                with open(f_file, 'r') as f:
                    try:
                        frame_data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    
                if len(frame_data['instance_info']) == 1:
                    instance = frame_data['instance_info'][0] 
                
                    kp = np.array(instance['keypoints'][5:23])
                    assert kp.shape == (18, 2), f"Expected keypoints to have shape (18, 2), but got {kp.shape} in {f_file}"
                    scores = np.array(instance['keypoint_scores'][5:23])
                    assert scores.shape == (18,), f"Expected scores to have shape (18, 1), but got {scores.shape} in {f_file}"
                    obj_id = instance.get('obj_id', -1)
                
                elif len(frame_data['instance_info']) > 1:
                    raise ValueError(f"Multiple instances found in {f_file}, expected only one. Please check the data format.")
                
                else:
                    print(f"Warning: No instances found in {f_file}, skip next all frames.")
                    break
                    
                sequence_keypoints.append(kp)
                sequence_scores.append(scores)
                sequence_obj_ids.append(obj_id)
                
            assert len(sequence_keypoints) == len(sequence_scores), f"Data length mismatch in {vid_path}"
            dataset_records[vid_dir] = {
                'keypoints': np.stack(sequence_keypoints),         # Shape: (T, 18, 2)
                'keypoint_scores': np.stack(sequence_scores),      # Shape: (T, 18)
                'obj_ids': np.array(sequence_obj_ids),             # Shape: (T,)
                'total_frames': len(sequence_keypoints),      
            }
                
        print(f"Processed {len(dataset_records)} test patients data.")
        print(f"Saving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(dataset_records, f)
        print("Test datasets saved completely!")
    

if __name__ == "__main__":
    TRACK = 'track2'  
    DATASET_DIR = "dataset"
    LABEL_FILE = f"dataset/gt/{TRACK}_train.json"
    SPLIT_FILE = f"dataset/dataset_split_{TRACK}.pt"
    TRAIN_OUTPUT_FILE = f"dataset/train_dataset_{TRACK}.pkl"
    VAL_OUTPUT_FILE = f"dataset/val_dataset_{TRACK}.pkl"
    
    process_train_datasets(DATASET_DIR, LABEL_FILE, TRAIN_OUTPUT_FILE, VAL_OUTPUT_FILE, split_file=None)


    # track1_list = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
    # track2_list = [4, 6, 7, 13, 26, 35, 39, 42, 50]
    # process_test_datasets(DATASET_DIR, track1_list)
