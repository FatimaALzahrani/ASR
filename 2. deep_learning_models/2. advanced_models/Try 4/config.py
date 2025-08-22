from typing import Dict


def get_config(config_name: str = "basic") -> Dict:
    configs = {
        "basic": {
            "batch_size": 2,
            "learning_rate": 3e-5,
            "weight_decay": 1e-6,
            "epochs": 15,
            "patience": 5,
            "gradient_clipping": 1.0
        },
        "intermediate": {
            "batch_size": 4,
            "learning_rate": 5e-5,
            "weight_decay": 1e-5,
            "epochs": 25,
            "patience": 7,
            "gradient_clipping": 1.0
        },
        "advanced": {
            "batch_size": 6,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 40,
            "patience": 10,
            "gradient_clipping": 0.5
        }
    }
    
    return configs.get(config_name, configs["basic"])