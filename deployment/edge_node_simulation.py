import random
import time

class EdgeNode:
    def __init__(self, name, latency):
        self.name = name
        self.latency = latency

    def run_inference(self):
        time.sleep(self.latency)
        return random.random()


def simulate_network():
    nodes = [
        EdgeNode("pi_node", 0.2),
        EdgeNode("jetson_node", 0.1),
        EdgeNode("desktop_node", 0.05),
    ]

    results = {}

    for node in nodes:
        output = node.run_inference()
        results[node.name] = output

    print("🌐 Distributed Inference Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    simulate_network()