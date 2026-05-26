# Acknowledgments and Upstream Work

This repository builds on prior work in social robot navigation and pedestrian simulation. The
references below preserve the provenance that previously lived in the root `README.md`.

## Research lineage

Caruso, Matteo, Enrico Regolin, Federico Julian Camerota Verdu, Stefano Alberto Russo, Luca
Bortolussi, and Stefano Seriani. "Robot Navigation in Crowded Environments: A Reinforcement
Learning Approach." *Machines* 11, no. 2 (2023): 268.
<https://doi.org/10.3390/machines11020268>

As stated in the paper's data-availability statement, related public material was made available in:

- <https://github.com/EnricoReg/robot-sf>
- <https://github.com/EnricoReg/asynch-rl>
- <https://github.com/matteocaruso1993/crowd_nav_experimental>

## Related upstream repositories

Additional repositories referenced in this project's lineage and implementation context:

- <https://github.com/Bonifatius94/robot-sf>
- <https://github.com/yuxiang-gao/PySocialForce>
- <https://github.com/Bonifatius94/PySocialForce>

## fast-pysf / PySocialForce lineage

The `fast-pysf/` subtree and surrounding pedestrian-simulation work acknowledge the upstream model
and implementation lineage:

- Based on Sven Kreiss's implementation of the vanilla Social Force model:
  <https://github.com/svenkreiss/socialforce>
- Force-implementation details also drew inspiration from:
  <https://github.com/srl-freiburg/pedsim_ros>

## Core references for the pedestrian model

- Helbing, D., and P. Molnar. "Social force model for pedestrian dynamics." *Physical Review E* 51,
  no. 5 (1995): 4282-4286. <https://doi.org/10.1103/PhysRevE.51.4282>
- Moussaid, M., N. Perozo, S. Garnier, D. Helbing, and G. Theraulaz. "The walking behaviour of
  pedestrian social groups and its impact on crowd dynamics." *PLoS ONE* 5, no. 4 (2010): 1-7.
  <https://doi.org/10.1371/journal.pone.0010047>
