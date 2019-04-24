import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Train a neural net to solve the online k-server problem')
    parser.add_argument("--model", "-m", type=str, choices=['lstm', 'fc'], default='lstm',
                        help="pick the model to train with")
    parser.add_argument("--epochs", "-e", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.1,
                        help="Learning rate of the algorithm")
    parser.add_argument("--optim", "-op", type=str, choices=['adam', 'sgd'], default='sgd',
                        help="Choice of optimization algorithm")    
    parser.add_argument("--n_servers", "-s", type=int, default=10,
                        help="number of servers in the instance")
    parser.add_argument("--n_requests_train", "-r", type=int, default=100000,
                        help="number of requests to train with")
    parser.add_argument("--batch_size", "-b", type=int, default=100,
                        help="number of requests in each batch to train with")
    parser.add_argument("--n_requests_test", "-rt", type=int, default=1000,
                        help="number of requests to test with overall")
    parser.add_argument("--hidden_layers", "-hl", type=int, default=3,
                        help="number of hidden layers in the network")
    parser.add_argument("--hidden_units", "-hu", type=int, default=10,
                        help="number of hidden units in each layer of the network")
    parser.add_argument("--dims", "-d", type=int, default=2,
                        help="Dimension of the dataset")
    parser.add_argument("--dist_metric", "-dm", type=int, default=2,
                        help="Mertic in which the distance is measured")
    parser.add_argument("--model_dir", type=str, default='trained_models/lstm',
                        help="Directory where we should save the model and the tensorboard")
    return parser