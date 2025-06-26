import flwr as fl
import copy
import torch
import numpy as np
from pneumonia_detector.trainer import train_model, evaluate_model

def start_fl_server():
    fl.server.start_server(config={"num_rounds": 3})

def start_fl_client(model, train_fn):
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config): return [val.cpu().numpy() for val in model.parameters()]
        def fit(self, parameters, config): pass  # TODO: implement
        def evaluate(self, parameters, config): pass
    fl.client.start_numpy_client
    
def average_state_dicts(state_dicts):
    """
    Given a list of state_dicts, return the element-wise average.
    """
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        for sd in state_dicts[1:]:
            avg[key] += sd[key]
        avg[key] = torch.div(avg[key], len(state_dicts))
    return avg

def run_federated_simulation(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    num_clients: int,
    num_rounds: int,
    strategy: str,
    epochs_per_round: int,
    learning_rate: float,
    writer=None,
    global_step=0,
    patience=5
):
    """
    Very simple FedAvg simulator:
      - Each round, we clone the global model for each client,
        train locally (same full dataset, for demo), collect weights,
        average them, and update global model.
      - Finally evaluate on test_loader.
    """
    global_model = copy.deepcopy(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = global_step

    for rnd in range(1, num_rounds + 1):
        print(f"\n=== Federated Round {rnd}/{num_rounds} ===")
        client_states = []
        for cid in range(num_clients):
            local_model = copy.deepcopy(global_model)
            print(f"[Client {cid+1}] training locally …")
            local_model, step = train_model(
                model=local_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_round,
                learning_rate=learning_rate,
                device=device,
                writer=writer,
                global_step=step,
                patience=patience
            )
            client_states.append(local_model.state_dict())

        # FedAvg
        print("Averaging client models …")
        new_state = average_state_dicts(client_states)
        global_model.load_state_dict(new_state)
        
        if writer:
            writer.add_scalar("FL/Round", rnd, step)

    # Final evaluation
    print("\n=== Final Evaluation on Test Set ===")
    evaluate_model(global_model, test_loader, device=device, tag="Test")

    return global_model