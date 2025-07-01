# common/data_utils.py - Updated to ensure all 52 features

import numpy as np
import os

def load_client_data(data_path):
    """Loads all .npy files for a client and assigns labels."""
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    print(f"[Data] Loading from: {os.path.abspath(data_path)}")
    
    # Load normal data (class 0)
    try:
        normal_train = np.load(os.path.join(data_path, 'normal_train.npy'))
        normal_test = np.load(os.path.join(data_path, 'normal_test.npy'))
        
        X_train_list.append(normal_train)
        y_train_list.append(np.full(normal_train.shape[0], 0))
        X_test_list.append(normal_test)
        y_test_list.append(np.full(normal_test.shape[0], 0))
        
        print(f"[Data] Normal data - Train: {normal_train.shape}, Test: {normal_test.shape}")
        
    except FileNotFoundError:
        print("[Data ERROR] Normal data files not found!")
        return None, None, None, None

    # Load fault data (classes 1-21)
    fault_count = 0
    for i in range(1, 22):
        fault_id = f'fault_{i:02d}'
        train_file = os.path.join(data_path, f'{fault_id}_train.npy')
        test_file = os.path.join(data_path, f'{fault_id}_test.npy')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            fault_train = np.load(train_file)
            fault_test = np.load(test_file)
            
            X_train_list.append(fault_train)
            y_train_list.append(np.full(fault_train.shape[0], i))
            X_test_list.append(fault_test)
            y_test_list.append(np.full(fault_test.shape[0], i))
            
            fault_count += 1
    
    print(f"[Data] Loaded {fault_count} fault types")
    
    # Combine all data
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    print(f"[Data] Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")

    if X_train.shape[1] != 52:
        print(f"[Data WARNING] Expected 52 features, got {X_train.shape[1]}")
    else:
        print(f"[Data] Confirmed: Using all 52 TEP features (BALANCED DATASET)")
    
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"[Data] BALANCED Training class distribution: {class_dist}")
    total_samples = sum(counts)
    print(f"[Data] Total training samples: {total_samples}, Classes: {len(unique)}")
    
    return X_train, y_train, X_test, y_test

def create_sequences(X, y, time_steps=20):
    """Creates overlapping sequences for time-series learning."""
    if len(X) <= time_steps:
        print(f"[Data WARNING] Not enough data points ({len(X)}) for time_steps ({time_steps})")
        return np.array([]), np.array([])
    
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        # Take a window of time_steps
        sequence = X[i:(i + time_steps)]
        label = y[i + time_steps]  # Label for the sequence is the next time step
        
        Xs.append(sequence)
        ys.append(label)
    
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    
    print(f"[Data] Created {len(X_seq)} sequences of shape {X_seq.shape[1:]} -> {y_seq.shape}")
    return X_seq, y_seq