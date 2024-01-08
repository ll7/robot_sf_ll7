# PySocialForce

[](https://github.com/Bonifatius94/PySocialForce/actions/workflows/ci.yml/badge.svg)

## About
This project is a Python implementation of the **Extended Social Force Model** [[2]](#2).
It extends the vanilla social force model [[1]](#1) to simulate the walking behaviour
of pedestrians with social group interactions.

The pedestrian states and other simulated entities are represented by NumPy arrays.
Performance-critical implementations of forces use Numba for significant speed-ups.

## Installation

Clone the source code.

```sh
git clone https://github.com/Bonifatius94/PySocialForce.git
```

Install the pysocialforce package and its dependencies using pip + setup.py.

```sh
pip install -e .
```

For development, run the automated tests and lint the coding style.

```sh
pylint pysocialforce
pytest tests
```

## Usage
See the usage examples in the [examples](./examples/) folder.

## License
This project is available under the MIT License.

## References

<a id="1">[1]</a> Helbing, D., & Molnár, P. (1995). Social force model
for pedestrian dynamics. Physical Review E, 51(5), 4282–4286.
<https://doi.org/10.1103/PhysRevE.51.4282>

<a id="2">[2]</a> Moussaïd, M., Perozo, N., Garnier, S., Helbing, D., & Theraulaz, G. (2010).
The walking behaviour of pedestrian social groups and its impact on crowd dynamics.
PLoS ONE, 5(4), 1–7. <https://doi.org/10.1371/journal.pone.0010047>

<a id="3">[3]</a> Sven Kreiss's original Social Force implementation
on [GitHub](https://github.com/svenkreiss/socialforce)

<a id="4">[4]</a> pedsim_ros implementation on [GitHub](https://github.com/srl-freiburg/pedsim_ros)
