# Client_1/Client.py and Client_2/Client.py - Definitive version with a working "Retry" loop

import flwr as fl
import numpy as np
import argparse
import tensorflow as tf
import threading
import time
import json
import sys
import os
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.llm_interface import load_local_llm, get_llm_response
from common.model import create_cnn_lstm_attention_model, create_simple_cnn_lstm_model
from common.data_utils import load_client_data, create_sequences
from common.tep_variables import TEP_VARIABLES

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ImprovedClientAgent:
    """ Manages the client's state and user interaction. """
    def __init__(self, client_id, data_path):
        self.client_id = client_id
        self.data_path = data_path
        self.model = None
        self.X_train_seq, self.y_train_seq = None, None
        self.X_test_seq, self.y_test_seq = None, None
        self.class_weights = None
        self.last_used_config = None
        self.scaling_plan = None
        
        # --- NEW, SIMPLIFIED & ROBUST THREADING LOGIC ---
        # This channel is used by the background thread to request user input.
        self.user_prompt_event = threading.Event()
        self.prompt_text_for_user = ""
        self.user_response = ""

    def _get_llm_scaling_plan(self, training_intent):
        """Asks the LLM to fill in a JSON template for the scaling plan."""
        print(f"[Client {self.client_id}] LLM is deciding on a scaling strategy for task: '{training_intent}'...")
        
        variable_types = sorted(list(set(v['type'] for v in TEP_VARIABLES.values())))
        json_template = "{\n" + ",\n".join([f'  "{v_type}": "<decision>"' for v_type in variable_types]) + "\n}"

        if training_intent == 'precision':
            system_message = f"""
            You are a data scaling expert for a high-precision monitoring task. Your goal is to fill in a JSON template with your scaling decisions.
            The available scaling methods are: 'standard', 'minmax', or 'none'.
            For this precision task, it is vital to preserve the original data distribution of variables like 'pressure' and 'composition'. Therefore, you should choose 'none' for them. Scale other less critical types.
            """
        else: # 'general'
            system_message = f"""
            You are a data scaling expert for a general fault diagnosis task. Your goal is to fill in a JSON template.
            The available scaling methods are: 'standard', 'minmax', or 'none'.
            Based on general principles: use 'standard' for normally distributed types like 'pressure' and 'temperature'. Use 'minmax' for types with a fixed range like 'level'.
            """
        
        prompt = f"Here is the JSON template you MUST fill out. Do not change the keys. Replace each '<decision>' with your chosen scaling method ('standard', 'minmax', or 'none').\n\nTemplate:\n{json_template}\n\nYour completed JSON:"
        
        try:
            scaling_plan_json, raw_text = get_llm_response(prompt, system_message)
            # We add a check to ensure the LLM didn't just return the template
            if scaling_plan_json and all(k in variable_types for k in scaling_plan_json.keys()) and all(v != '<decision>' for v in scaling_plan_json.values()):
                print(f"[Client {self.client_id}] LLM successfully created a scaling plan.")
                self.scaling_plan = scaling_plan_json
                return True
            else:
                print(f"[Client {self.client_id} INFO] LLM returned an invalid or incomplete plan. Raw output: {raw_text}")
                raise ValueError("LLM response did not match the required template structure or was incomplete.")
        except Exception as e:
            print(f"[Client {self.client_id} WARNING] LLM scaling task failed ({e}).")
            self.scaling_plan = None
            return False

    def _prepare_data(self, server_config):
        """Prepares the data based on the approved configuration."""
        print(f"[Client {self.client_id}] Preparing data for config: {server_config}")
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_client_data(self.data_path)
        if X_train_raw is None: sys.exit(1)

        scaling_strategy = server_config.get("scaling_strategy", "none")
        preprocessor = None
        
        if scaling_strategy == "llm_smart_scale" and self.scaling_plan:
            print(f"[Client {self.client_id}] Applying user-approved LLM scaling plan...")
            transformers, scaler_groups = [], {'standard': [], 'minmax': []}
            for i in range(1, 53):
                var_type = TEP_VARIABLES.get(i, {}).get('type')
                scaler_choice = self.scaling_plan.get(var_type)
                if scaler_choice in ['standard', 'minmax']:
                    scaler_groups[scaler_choice].append(i-1)
            
            if scaler_groups['standard']: transformers.append(('standard', StandardScaler(), scaler_groups['standard']))
            if scaler_groups['minmax']: transformers.append(('minmax', MinMaxScaler(), scaler_groups['minmax']))
            
            if transformers: preprocessor = ColumnTransformer(transformers, remainder='passthrough', sparse_threshold=0)
            else: print(f"[Client {self.client_id}] LLM chose not to scale any features.")
        elif scaling_strategy == "standard" or (scaling_strategy == "llm_smart_scale" and not self.scaling_plan):
             if scaling_strategy == "llm_smart_scale":
                 print(f"[Client {self.client_id}] No valid LLM plan approved. Applying StandardScaler as a fallback.")
             else:
                 print(f"[Client {self.client_id}] Applying StandardScaler to all features.")
             preprocessor = StandardScaler()

        if preprocessor:
            X_train_scaled = preprocessor.fit_transform(X_train_raw)
            X_test_scaled = preprocessor.transform(X_test_raw)
        else: # 'none'
            print(f"[Client {self.client_id}] No scaling applied.")
            X_train_scaled, X_test_scaled = X_train_raw, X_test_raw

        window_size = 20
        self.X_train_seq, self.y_train_seq = create_sequences(X_train_scaled, y_train_raw, time_steps=window_size)
        self.X_test_seq, self.y_test_seq = create_sequences(X_test_scaled, y_test_raw, time_steps=window_size)
        unique_classes = np.unique(self.y_train_seq)
        weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=self.y_train_seq)
        self.class_weights = dict(zip(unique_classes, weights))
        self.model = None 
        print(f"[Client {self.client_id}] Data ready.")

    def _initialize_model(self, model_type="cnn_lstm_attention"):
        """Initializes the ML model."""
        if self.model is None:
            print(f"[Client {self.client_id}] Creating {model_type} model...")
            input_shape = (self.X_train_seq.shape[1], self.X_train_seq.shape[2])
            if model_type == "cnn_lstm_attention": self.model = create_cnn_lstm_attention_model(input_shape=input_shape, num_classes=22)
            else: self.model = create_simple_cnn_lstm_model(input_shape=input_shape, num_classes=22)

    def get_user_input(self, prompt):
        """This function is called by the background thread to safely get input from the main thread."""
        self.prompt_text_for_user = prompt
        self.user_prompt_event.set() # Signal the main thread that we need input
        # Now, we wait until the main thread clears the event, which means it has placed a response
        while self.user_prompt_event.is_set():
            time.sleep(0.2)
        return self.user_response

    def main_menu_loop(self):
        """The main thread loop, dedicated to handling user input when requested."""
        print(f"\n=== Client {self.client_id} Ready ==="); print("Waiting for training requests...")
        while True:
            # This loop's only job is to wait for the signal to ask a question
            if self.user_prompt_event.is_set():
                response = input(self.prompt_text_for_user).lower().strip()
                self.user_response = response
                self.user_prompt_event.clear() # Signal back that the response is ready
            time.sleep(0.1)

class ImprovedTEPClient(fl.client.NumPyClient):
    def __init__(self, agent: ImprovedClientAgent):
        self.agent = agent

    def get_parameters(self, config):
        """Called by the server to get initial model parameters."""
        print(f"[Client {self.agent.client_id}] Server requesting parameters...")
        if self.agent.model is None:
            temp_config = {"model_type": "cnn_lstm_attention", "scaling_strategy": "standard"}
            self.agent._prepare_data(temp_config)
            self.agent._initialize_model(temp_config.get("model_type"))
            self.agent.last_used_config = temp_config
        return self.agent.model.get_weights()

    def fit(self, parameters, config):
        """This is the main training method called by the Flower server."""
        server_round = config.get("server_round", 1)
        
        # --- "ASK ONCE" LOGIC for the whole run ---
        if server_round == 1:
            response = self.agent.get_user_input(f"\n*** TRAINING REQUEST RECEIVED ***\nStart training run? (yes/no): ")
            if response not in ['yes', 'y']:
                print(f"[Client {self.agent.client_id}] Training declined by user.")
                return self.agent.model.get_weights(), 0, {"status": "declined"}
            print(f"[Client {self.agent.client_id}] Training approved.")
        
        # --- EFFICIENCY LOGIC & "RETRY" WORKFLOW ---
        core_config = {k: v for k, v in config.items() if k != 'server_round'}
        if self.agent.last_used_config != core_config:
            if config.get("scaling_strategy") == "llm_smart_scale":
                while True: # The "Retry" loop
                    llm_success = self.agent._get_llm_scaling_plan(config.get('training_intent', 'general'))
                    
                    if llm_success:
                        print("\n--- LLM SCALING PLAN REVIEW ---")
                        print("The LLM has proposed the following scaling plan:")
                        print(json.dumps(self.agent.scaling_plan, indent=2))
                        response = self.agent.get_user_input("Approve this scaling plan? (yes/no/retry): ")
                    else:
                        response = self.agent.get_user_input("[Agent] LLM call failed to produce a valid plan. Retry? (yes/no): ")
                        if response in ['yes', 'y']: response = 'retry' # Treat 'yes' as 'retry' on failure
                    
                    if response == 'retry':
                        print("[Agent] User requested retry. Asking LLM again...")
                        continue # Go back to the start of the while loop and re-run LLM
                    elif response in ['yes', 'y']:
                        print("[Agent] Scaling plan approved.")
                        break # Exit the loop and proceed with this plan
                    else: # 'no' or anything else
                        print("[Agent] Scaling plan declined. Using fallback.")
                        self.agent.scaling_plan = None # Ensure plan is empty
                        break # Exit the loop and proceed with fallback

            self.agent._prepare_data(config)
            self.agent.last_used_config = core_config
        else:
             print(f"\n[Client {self.agent.client_id}] Round {server_round} - Config unchanged, using cached data.")
        
        self.agent._initialize_model(config.get("model_type", "cnn_lstm_attention"))
        print(f"[Client {self.agent.client_id}] Round {server_round} - Training started...")
        self.agent.model.set_weights(parameters)
        
        callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=3),
                      EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) ]
        
        history = self.agent.model.fit(
            self.agent.X_train_seq, self.agent.y_train_seq,
            epochs=20, batch_size=32, validation_split=0.15,
            class_weight=self.agent.class_weights,
            callbacks=callbacks, verbose=0
        )
        final_val_accuracy = max(history.history.get('val_accuracy', [0]))
        print(f"[Client {self.agent.client_id}] Round {server_round} - Training complete. Val Acc: {final_val_accuracy:.3f}")
        return self.agent.model.get_weights(), len(self.agent.X_train_seq), {"val_accuracy": final_val_accuracy}

    def evaluate(self, parameters, config):
        server_round = config.get("server_round", "N/A")
        print(f"[Client {self.agent.client_id}] Round {server_round} - Evaluating...")
        if self.agent.model is None: return 0.0, 0, {"accuracy": 0.0}
        self.agent.model.set_weights(parameters)
        loss, accuracy = self.agent.model.evaluate(self.agent.X_test_seq, self.agent.y_test_seq, verbose=0)
        print(f"[Client {self.agent.client_id}] Round {server_round} - Test accuracy: {accuracy:.3f}")
        return loss, len(self.agent.X_test_seq), {"accuracy": accuracy}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True, choices=[1, 2])
    args = parser.parse_args()
    print(f"=== Starting Client {args.client_id} ===")
    data_path = "local_data/"
    agent = ImprovedClientAgent(client_id=args.client_id, data_path=data_path)
    flower_client = ImprovedTEPClient(agent)
    def start_fl_client():
        fl.client.start_client(server_address="0.0.0.0:8080", client=flower_client.to_client())
    fl_thread = threading.Thread(target=start_fl_client)
    fl_thread.daemon = True
    fl_thread.start()
    try:
        agent.main_menu_loop()
    except KeyboardInterrupt:
        print(f"\n[Client {args.client_id}] Shutting down")
    