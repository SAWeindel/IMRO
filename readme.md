This Repository contains the Tech Report and Implementation accompanying the publication

# Interaction-Minimizing Roadmap Optimization for High-Density Multi-Agent Path Finding

Published in the [2025 IEEE 21st International Conference on Automation Science and Engineering (CASE)](https://ieeexplore.ieee.org/xpl/conhome/11163731/proceeding), available under DOI [CASE58245.2025.11163822](https://doi.org/10.1109/CASE58245.2025.11163822).

The [Tech Report](Tech_Report.pdf) is based on the Master’s Thesis **Planning Magnetic Levitation Operating Strategies for PLC-Controlled Intralogistics**, which is tailored to the specific requirements of the [CORAS](https://www.somic-packaging.com/en/solutions/collating-and-grouping-systems/coras.html) collecting and grouping system by SOMIC Verpackungsmaschinen GmbH & Co. KG. The provided [Implementation](./implementation/) has been used to generate the presented data.

# Abstract
Modern industry increasingly demands customizability from each element of their workflow and factories. A prominent example for this is the advent of Automated Guided Vehicles (AGV) in intralogistics tasks, which autonomously navigate the manufacturing floor, reacting dynamically to variations in the workflow. One such application makes use of magnetically propelled planar drive systems to transport products between manufacturing stations, replacing traditional solutions which are limited in their ability to efficiently adapt to new requirements.This work presents an AGV control approach capable of offloading large parts of the computational expense into an offline preprocessing step: A unidirectional roadmap is generated using alternating position optimization and network modification operations, with the goal of reducing the number of interactions between agents to be resolved at runtime. This concept was successfully validated in simulation. An accompanying tech report and implementation further details the presented approach, as well as the used controller and simulator.

![Pipeline of the presented approach](Pipeline.svg)

# License
Copyright 2025 Sören Arne Weindel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use the contents of this repository except in compliance
with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.