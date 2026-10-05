# LoRaFL

A Python simulator for federated learning over a shared LoRaWAN-inspired
random-access uplink. It models compressed client updates, fragmentation,
collisions, duty-cycle timing, aggregation, and uplink transmit cost.

## Installation

Python 3.12 is recommended.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

```sh
python newmnistlora.py --help
```

Datasets are downloaded through Flower Datasets; the first run requires
internet access. Simulation output is written to `results/`.

## Tests

```sh
python -m unittest discover -s tests -v
```

The included tests cover channel allocation, fragment timing, and duty-cycle
behavior. They do not validate all learning behavior.

## Model scope

This is a LoRaWAN-inspired simulation, not a standards-complete implementation.
The channel uses overlap collisions. Downlink model delivery is reliable and
its timing is modeled separately. Uplink transmission costs exclude device
computation, receive and idle energy, and gateway/server energy.

## Reproducibility

This package currently provides the simulation core. Paper-specific sweep
launchers, configurations, and required reference artifacts are not yet part
of this public package; it is not yet a complete paper reproduction release.
