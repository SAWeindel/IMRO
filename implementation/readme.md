# Setup
The Implementation has been tested using Python 3.12.3 and the packages specified in the [requirements.txt](./requirements.txt). The working directory must be set to [IMRO/implementation](./).

A [demonstration](./demo.ipynb) file is provided to outline the structure of the approach. The [main](./main.py) file is streamlined for more extensive use.

# Directory Structure
| Directory | Content |
| ----------- | --------- |
| [definition](definition/) | Contains all file structures used by the approach. |
| [offline](offline/) | Contains all workers of the offline preprocessing pipeline. The entry point is [planner.py](offline/planner.py). |
| [online](online/) | Contains the online [simulator](online/simulator.py) and [controller](online/controller.py). |
| [tools](tools/) | Contains logging and interface tools. |
| [tasks](tasks/) | Contains exemplary environment files. |
| [output](output/) | Target for the output files of the approach. |
| [runs](runs/) | Target of the archiving function of the [main](./implementation/main.py) function. While `save_run` is True, each execution is logged there for reference. |

# Environment .json Structure
Environments are contained in files at `tasks/env_{name}.json`. They contain the problem statement and serve as input to the offline preprocessing algorithm. The .json files are structured as follows:

| 1st level   | 2nd level | Valid entries    | Description |
| ----------- | --------- | ---------------- | ----------- |
| `grid`      | `dims`    | `[num, num]`     | Number of tiles forming the field of the environment. |
| `grid`      | `size`    | `[num, num]`     | Size of each tile, in mm. |
| `movers`    | `size`    | `[num, num]`     | Size of each mover, in mm. |
| `pos_units` |           | `("grid", "mm")` | Reference for station positions. For `"grid"` the station is centered on the indexed tile, for `"mm"` the center is given as a point on the field. |
| `providers` |           | `[entry, ...]`   | Specifies the provider stations placed on the field. At least one provider is required. |
| `receivers` |           | `[entry, ...]`   | Specifies the receiver stations placed on the field. At least one receiver is required. |

Station entries are structured as follows:

| Key         | Valid entries         | Description |
| ----------- | --------------------- | ----------- |
| `pos`       | `[num, num]`          | Position of the center of the station, as defined by `pos_units`. |
| `size`      | `[num, num]`            | Size of the station in mm. |
| `items`     | `[num, ...]`          | Ordered sequence of item types provided/received by the station. Item types may repeat within and between stations, with `0` being reserved for the empty item type. |
| `item_type` | `("any", "sequence")` | Placeholder for non-sequence item type requirements. |
| `period`    | `num`                 | Duration between the mean required time of two interactions, in seconds. |
| `duration`  | `num`                 | Mean duration of an interaction, in seconds. |

# Output Files
The approach by default generates a number of files, placed in the [output](output/) directory.

| Name | Content |
| ----------- | --------- |
| [log.txt](output/log.txt) | Updated during execution. Contains extensive information about the state of the execution. |
| [nodes.txt](output/nodes.txt) | Readable version of the final roadmap graph. |
| [events.txt](output/events.txt) | Log of station - mover interaction events during the online simulation. |
| [actions.txt](output/actions.txt) | Log of mover actions during the online simulation. |
| [simulation.txt](output/simulation.txt) | Updated during execution. Log of controller actions during the online simulation. |
| {}.png | Visualizations of the Network.
| sol_{}.json | Networks at different stages of the pipeline. Can be modified or used as continuation points.
| iter_{}.json | Checkpoint networks generated during the placement optimization. Can be modified or used as continuation points.
| [force_placement.avi](output/force_placement.avi) | Animated optimization of the network. |
| [controller_sequence.avi](output/controller_sequence.avi) | Animated output of the online simulation. |
| [sol.p](output/sol.p) | Pickled solution of the offline pipeline. Can used as a starting point for the online simulation or visualized. |
| [seq.p](output/seq.p) | Pickled sequence resulting from the the online simulation. Can be visualized. |