# Fed-Learning-llama

## Steps to Run

1. Clone the DATASET repository inside the `Data` folder:

```bash
git clone https://github.com/YKatser/CPDE
```

2. Create virtual environments inside each of the `client` and `server` folders and install the requirements:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the model file `tinyllama-1.1b-chat-v1.0.Q2_K.gguf` from Hugging Face and place it inside the `Models` folder.

4. To run the system, open 3 separate terminals and activate the virtual environments inside each.

5. Run the server first, then both clients:

```bash
# Terminal 1 - Server
cd Server
source venv/bin/activate
python server.py

# Terminal 2 - Client 1
cd Client1
source venv/bin/activate
python client.py

# Terminal 3 - Client 2
cd Client2
source venv/bin/activate
python client.py
```
