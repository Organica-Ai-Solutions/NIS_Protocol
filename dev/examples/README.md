# NIS Protocol Examples

This directory contains example implementations of the NIS Protocol for various use cases.

## Examples

- **Automated Tolling System**: Complete example of a toll booth system using computer vision and NIS Protocol agents.
- **Smart Traffic monitoring ([health tracking](src/infrastructure/integration_coordinator.py))**: Example implementation of traffic analysis with emotional weighting.
- **Basic Agent Communication**: Simple example showing how agents communicate and make decisions.

## Running the Examples

Each example has its own directory with a README explaining how to run it.

General steps:

1. Ensure Redis is running:
```bash
redis-server
```

2. Install dependencies:
```bash
pip install -e ".[vision]"
```

3. Run the specific example:
```bash
python examples/basic_agent_communication/run.py
```

## Creating Your Own Examples

You can use these examples as a starting point for your own NIS Protocol implementations.

1. Create a new directory for your example
2. Implement the necessary agents using the NIS Protocol framework
3. Create a run.py file to demonstrate your example
4. Submit a pull request if you'd like to contribute your example to the project 