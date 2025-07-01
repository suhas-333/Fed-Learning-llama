# Server/server.py - Final version for Task-Specific LLM Experiments

import flwr as fl
import json
import threading
import time
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.llm_interface import load_local_llm

class ServerAgent:
    def __init__(self, num_expected_clients=2):
        print("=== TEP Federated Learning Server ===")
        self.num_expected_clients = num_expected_clients
        self.trained_models_archive = {}
        self.is_training = False
        self.fl_thread = None
        self.connected_clients = 0
        
        print("Checking for LLM model...")
        load_local_llm()
        print("Server ready!")

    def _guided_training_setup(self):
        """A guided menu to configure the training run, including the high-level intent."""
        print("\n=== Training Configuration ===")
        config = {}
        
        # Step 1: Define the high-level goal for the LLM
        print("1. Choose the Training Intent:")
        print("   - general: For robust, all-around fault diagnosis (standard behavior).")
        print("   - precision: For precision monitoring, preserving original data distributions where possible.")
        while True:
            choice = input("   Choose intent ('general' or 'precision'): ").lower().strip()
            if choice in ["general", "precision"]:
                config["training_intent"] = choice
                break
            else:
                print("   Invalid choice.")

        # Step 2: Model type selection
        print("\n2. Available models:\n   - cnn_lstm_attention (research-proven, recommended)\n   - simple_cnn_lstm (simpler version)")
        while True:
            choice = input("   Choose model (1 or 2): ").strip()
            if choice == "1":
                config["model_type"] = "cnn_lstm_attention"; break
            elif choice == "2":
                config["model_type"] = "simple_cnn_lstm"; break
            else:
                print("   Invalid choice. Enter 1 or 2.")
        
        # Step 3: Scaling strategy selection
        print("\n3. Choose Scaling Strategy:")
        print("   - none: No feature scaling will be applied.")
        print("   - standard: Apply StandardScaler to all 52 features.")
        print("   - llm: Let each client's LLM decide the best scaler based on the chosen Training Intent.")
        while True:
            choice = input("   Choose scaling ('none', 'standard', 'llm'): ").lower().strip()
            if choice in ["none", "standard", "llm"]:
                config["scaling_strategy"] = "llm_smart_scale" if choice == "llm" else choice
                break
            else:
                print("   Invalid choice.")

        config["feature_selection_strategy"] = "all"
        
        print("\n--- Configuration Summary ---")
        print(f"  Intent: {config['training_intent']}")
        print(f"  Model: {config['model_type']}")
        print(f"  Scaling: {config['scaling_strategy']}")
        
        while True:
            confirm = input("\nStart training? (yes/no): ").lower().strip()
            if confirm in ["yes", "y"]: return config
            elif confirm in ["no", "n"]: print("Training cancelled"); return None
            else: print("Enter yes or no")

    def _start_fl_run_in_background(self, config):
        """Starts the Flower server in a background thread."""
        def configure_round(server_round: int):
            config['server_round'] = server_round
            return config

        def aggregate_fit_metrics(metrics):
            val_accuracies = [m[1].get("val_accuracy", 0.0) for m in metrics]
            return {"val_accuracy": np.mean(val_accuracies)}

        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0, 
            min_fit_clients=self.num_expected_clients,
            min_available_clients=self.num_expected_clients, 
            on_fit_config_fn=configure_round, 
            on_evaluate_config_fn=configure_round,
            fit_metrics_aggregation_fn=aggregate_fit_metrics
        )

        def run_fl():
            self.is_training = True
            print("\n*** FEDERATED TRAINING STARTED ***")
            
            history = fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds=10),
                strategy=strategy,
            )
            
            model_name = f"model_{len(self.trained_models_archive) + 1}"
            final_accuracy = history.metrics_distributed.get("accuracy", [])
            self.trained_models_archive[model_name] = {
                "config": {k:v for k,v in config.items() if k != 'server_round'}, # Save config without round number
                "status": "Ready",
                "final_accuracy": final_accuracy[-1][1] if final_accuracy else "N/A"
            }
            print(f"\n*** TRAINING COMPLETE ***\nModel saved as: {model_name}")
            self.is_training = False

        self.fl_thread = threading.Thread(target=run_fl)
        self.fl_thread.start()

    def _list_trained_models(self):
        if not self.trained_models_archive: print("\nNo models trained yet")
        else:
            print("\n=== Trained Models ===")
            for name, details in self.trained_models_archive.items():
                print(f"  {name}: Intent={details['config']['training_intent']}, Model={details['config']['model_type']}, Scaling={details['config']['scaling_strategy']}, Final Accuracy={details.get('final_accuracy', 'N/A')}")
    
    def main_loop(self):
        while True:
            if self.is_training and self.fl_thread and self.fl_thread.is_alive(): self.fl_thread.join(); self.is_training = False
            
            print("\n=== TEP FL Server Menu ===")
            print(f"Status: {self.connected_clients}/{self.num_expected_clients} clients | Models: {len(self.trained_models_archive)}")
            print("1. Start Training\n2. List Models\n3. Exit")
            choice = input("Choice: ").strip()
            
            if choice == '1':
                config = self._guided_training_setup()
                if config: self._start_fl_run_in_background(config)
            elif choice == '2': self._list_trained_models()
            elif choice == '3': print("Goodbye!"); break
            else: print("Invalid choice")

if __name__ == "__main__":
    agent = ServerAgent()
    agent.main_loop()



