import numpy as np
from ortools.graph import pywrapgraph

class KServer:
    """
    A class that computes the optimal movement of k servers through a series of
    requests if the series of requests are known from beforehand.
    """
    def __init__(self, servers = None, requests = None, order = 0):
        """
        Initialize an instance with given list of servers, requests, and an order for the norm.
        """
        self.graph = None
        self.BigNumber = (1<<30) # Maximum weight it is able to handle
        self.dist_scaling = 1. * 10**5
        self.set_metric(order)
        self.add_servers(servers)
        self.add_requests(requests)

    def set_metric(self, order):
        """
        Set the order for the lp norm (recommended: 0, 1, 2, np.inf)
        """
        self.dist = lambda x, y: int(self.dist_scaling * np.linalg.norm(x - y, order))

    def add_servers(self, servers):
        """
        Add a list of servers to this problem instance.

        Parameters:
        servers: a n x d numpy array, where d is the dimension of the space and
        n is the number of servers.
        """
        self.servers = servers

    def add_requests(self, requests):
        """
        Add a list of requests to this problem instance.

        Parameters:
        requests: a r x d numpy array, where d is the dimension of the space and
        r is the number of requests. The requests should be in order.
        """
        self.requests = requests

    def optimal_cost(self):
        """
        Find the optimal cost (minimal distance traveled) by the k servers, 
        given the set of servers and the 
        """
        self._build_graph()
        if self.graph.Solve() == self.graph.OPTIMAL:
            return self._optimal_cost()
        else:
            raise ValueError('Problem in building and solving flow graph')

    def _optimal_cost(self):
        return (self.graph.OptimalCost() + (len(self.requests) * self.BigNumber)) / self.dist_scaling

    def _build_graph(self):
        self.graph = pywrapgraph.SimpleMinCostFlow()
        start_node = 0
        S = len(self.servers)
        R = len(self.requests)
        server_nodes = range(1, S + 1)
        # for every request, there are two nodes
        request_nodes = range(S + 1, S + 2 * R + 1, 2)
        end_node = S + 2 * R + 1

        # take a node number, and return the server location
        node_to_server = lambda x : self.servers[x - 1]
        # take a node number, and return the request location
        node_to_request = lambda x : self.requests[(x - 1 - S) // 2]

        # now build the graph
        for current_server_node in server_nodes:
            # Add the edges from source to servers
            self.graph.AddArcWithCapacityAndUnitCost(start_node, current_server_node, 1, 0)
            # And from server to sink
            self.graph.AddArcWithCapacityAndUnitCost(current_server_node, end_node, 1, 0)

            # now, for each server, add the cost to move from server to request.
            for current_request_node in request_nodes:
                self.graph.AddArcWithCapacityAndUnitCost(current_server_node, current_request_node, 1, 
                                                         self.dist(node_to_request(current_request_node), 
                                                                   node_to_server(current_server_node)))

        for current_request_node in request_nodes:
            # Add edge from request' to sink
            self.graph.AddArcWithCapacityAndUnitCost(current_request_node + 1, end_node, 1, 0)
            # Add edge from request to request' with a very low cost
            self.graph.AddArcWithCapacityAndUnitCost(current_request_node, current_request_node + 1, 1, -self.BigNumber)

            for next_request_node in range(current_request_node + 2, S + 2*R + 1, 2):
                # Add edge from request' to all next requests
                self.graph.AddArcWithCapacityAndUnitCost(current_request_node + 1, next_request_node, 1,
                                                         self.dist(node_to_request(current_request_node), 
                                                                   node_to_request(next_request_node)))

        # Add supply and demand at the start and end node
        self.graph.SetNodeSupply(start_node, S)
        self.graph.SetNodeSupply(end_node, -S)

