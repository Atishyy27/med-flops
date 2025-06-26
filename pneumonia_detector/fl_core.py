import flwr as fl

def start_fl_server():
    fl.server.start_server(config={"num_rounds": 3})

def start_fl_client(model, train_fn):
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config): return [val.cpu().numpy() for val in model.parameters()]
        def fit(self, parameters, config): pass  # TODO: implement
        def evaluate(self, parameters, config): pass
    fl.client.start_numpy_client("localhost:8080", client=Client())
