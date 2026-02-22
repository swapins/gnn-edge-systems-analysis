import uuid
import datetime

def create_experiment_metadata(config, dataset, hidden_dim, hardware_info):
    return {
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "dataset": dataset,
        "hidden_dim": hidden_dim,
        "hardware": hardware_info
    }