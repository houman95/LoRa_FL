"""Entry point for the compartmentalized LoRaWAN federated-learning simulator.

The implementation lives in the `lorawan_fl` package:

- `model.py`: neural network, training, testing, dataset loading
- `compression.py`: sparsification, quantization, update encoding
- `lora_phy.py` / `class_b.py`: LoRa airtime, MTU, Class-B downlink timing
- `access_policy.py`: duty-cycle-aware transmitters and collection policies
- `energy.py`: packet, airtime, and optional per-client/round energy budgets
- `channel.py`: ALOHA collision/success law
- `phases.py`: downlink, local training, uplink aggregation phases
- `experiment.py`: seed realizations, target metrics, and policy comparison runner
"""

from lorawan_fl.cli import main


if __name__ == "__main__":
    main()
