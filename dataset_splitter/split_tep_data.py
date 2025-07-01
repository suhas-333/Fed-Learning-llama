import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle

# --- Configuration ---
CPDE_DATA_PATH = '../data/CPDE/TEP_data/'
CLIENT1_DATA_PATH = '../Client_1/local_data/'
CLIENT2_DATA_PATH = '../Client_2/local_data/'

# TEP constants
NUM_TRAIN_SAMPLES_PER_FILE_EXPECTED = 480
NUM_TEST_SAMPLES_PER_FILE_EXPECTED = 960
NORMAL_SAMPLES_IN_FAULTY_TRAIN = 20
NORMAL_SAMPLES_IN_FAULTY_TEST = 160
NUM_VARIABLES = 52

# BALANCING STRATEGY
MAX_SAMPLES_PER_CLASS = 200  # Limit each class to max 200 samples per client
MIN_SAMPLES_PER_CLASS = 50   # Minimum samples needed per class

def load_tep_file(filepath, expected_rows):
    """Loads a .dat file into a NumPy array, handling transposed formats."""
    print(f"Loading: {filepath}")
    filename = os.path.basename(filepath)

    try:
        df = pd.read_csv(filepath, header=None, sep=r'\s+')
        data = df.to_numpy()

        # Handle transposed normal files
        if (filename in ['d00.dat', 'd00_te.dat']) and (data.shape[0] == NUM_VARIABLES):
            print(f"  Transposing {filename} from {data.shape}")
            data = data.T

        if data.shape[1] != NUM_VARIABLES:
            print(f"  ERROR: {filepath} has {data.shape[1]} columns, expected {NUM_VARIABLES}")
            return None

        print(f"  Loaded {filepath}: {data.shape}")
        return data

    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None

def balance_and_split_data():
    """Create balanced dataset with equal representation for all classes."""
    print("=== Creating BALANCED TEP Dataset ===")
    
    os.makedirs(CLIENT1_DATA_PATH, exist_ok=True)
    os.makedirs(CLIENT2_DATA_PATH, exist_ok=True)

    for data_type in ['train', 'test']:
        print(f"\n--- Processing {data_type}ing data ---")
        is_train = data_type == 'train'
        expected_rows = NUM_TRAIN_SAMPLES_PER_FILE_EXPECTED if is_train else NUM_TEST_SAMPLES_PER_FILE_EXPECTED
        
        all_class_data = {}  # Dictionary to store data for each class
        
        # --- Collect Normal Data (Class 0) ---
        print("Collecting normal data...")
        normal_file_prefix = 'd00_te' if not is_train else 'd00'
        normal_filepath = os.path.join(CPDE_DATA_PATH, normal_file_prefix + '.dat')
        
        normal_data_list = []
        
        # Main normal file
        if os.path.exists(normal_filepath):
            actual_expected = 500 if is_train else expected_rows
            normal_data = load_tep_file(normal_filepath, actual_expected)
            if normal_data is not None:
                normal_data_list.append(normal_data)
        
        # Normal portions from faulty files (use sparingly to avoid dominance)
        normal_from_faults = []
        for fault_num in range(1, 22):
            file_prefix = f'd{fault_num:02d}' + ('_te' if not is_train else '')
            filepath = os.path.join(CPDE_DATA_PATH, file_prefix + '.dat')
            
            if os.path.exists(filepath):
                fault_data = load_tep_file(filepath, expected_rows)
                if fault_data is not None:
                    normal_duration = NORMAL_SAMPLES_IN_FAULTY_TRAIN if is_train else NORMAL_SAMPLES_IN_FAULTY_TEST
                    normal_part = fault_data[:normal_duration, :]
                    normal_from_faults.append(normal_part)
        
        # Combine and limit normal data
        if normal_data_list:
            all_normal = np.concatenate(normal_data_list + normal_from_faults, axis=0)
        else:
            all_normal = np.concatenate(normal_from_faults, axis=0) if normal_from_faults else np.array([])
        
        # Balance normal data - don't let it dominate
        if len(all_normal) > MAX_SAMPLES_PER_CLASS * 2:  # *2 because we split between clients
            all_normal = shuffle(all_normal, random_state=42)[:MAX_SAMPLES_PER_CLASS * 2]
        
        all_class_data[0] = all_normal
        print(f"  Class 0 (Normal): {len(all_normal)} total samples")
        
        # --- Collect Fault Data (Classes 1-21) ---
        print("Collecting fault data...")
        for fault_num in range(1, 22):
            file_prefix = f'd{fault_num:02d}' + ('_te' if not is_train else '')
            filepath = os.path.join(CPDE_DATA_PATH, file_prefix + '.dat')
            
            if not os.path.exists(filepath):
                print(f"  WARNING: {filepath} not found, skipping fault {fault_num}")
                continue
            
            fault_file_data = load_tep_file(filepath, expected_rows)
            if fault_file_data is None:
                continue
            
            # Extract only the faulty portion
            fault_start = NORMAL_SAMPLES_IN_FAULTY_TRAIN if is_train else NORMAL_SAMPLES_IN_FAULTY_TEST
            if fault_file_data.shape[0] > fault_start:
                fault_data = fault_file_data[fault_start:, :]
                
                # Ensure we have enough samples
                if len(fault_data) < MIN_SAMPLES_PER_CLASS:
                    print(f"  WARNING: Fault {fault_num} has only {len(fault_data)} samples, may cause issues")
                
                all_class_data[fault_num] = fault_data
                print(f"  Class {fault_num} (Fault {fault_num}): {len(fault_data)} samples")
        
        # --- Balance All Classes ---
        print("\nBalancing classes...")
        min_samples = min(len(data) for data in all_class_data.values())
        target_samples = min(min_samples, MAX_SAMPLES_PER_CLASS * 2)  # *2 for both clients
        
        print(f"Target samples per class: {target_samples}")
        
        balanced_data = {}
        for class_id, data in all_class_data.items():
            if len(data) >= target_samples:
                # Randomly sample if we have more than needed
                shuffled = shuffle(data, random_state=42 + class_id)
                balanced_data[class_id] = shuffled[:target_samples]
            else:
                # Use all available data if less than target
                balanced_data[class_id] = data
            
            print(f"  Class {class_id}: {len(balanced_data[class_id])} samples after balancing")
        
        # --- Split Between Clients ---
        print("\nSplitting between clients...")
        
        for class_id, data in balanced_data.items():
            # Split approximately in half
            split_point = len(data) // 2
            client1_data = data[:split_point]
            client2_data = data[split_point:split_point*2]  # Take equal amount
            
            # Save data
            if class_id == 0:
                # Normal class
                filename = f'normal_{data_type}.npy'
            else:
                # Fault class
                filename = f'fault_{class_id:02d}_{data_type}.npy'
            
            if len(client1_data) > 0:
                np.save(os.path.join(CLIENT1_DATA_PATH, filename), client1_data)
                print(f"  Client 1 - Class {class_id}: {len(client1_data)} samples")
            
            if len(client2_data) > 0:
                np.save(os.path.join(CLIENT2_DATA_PATH, filename), client2_data)
                print(f"  Client 2 - Class {class_id}: {len(client2_data)} samples")
    
    print("\n=== BALANCED DATA SPLITTING COMPLETE ===")
    print(f"Data saved to:")
    print(f"  Client 1: {os.path.abspath(CLIENT1_DATA_PATH)}")
    print(f"  Client 2: {os.path.abspath(CLIENT2_DATA_PATH)}")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"- Each class limited to ~{MAX_SAMPLES_PER_CLASS} samples per client")
    print(f"- Normal class won't dominate the dataset")
    print(f"- All 22 classes (0 normal + 21 faults) should be balanced")

if __name__ == '__main__':
    balance_and_split_data()