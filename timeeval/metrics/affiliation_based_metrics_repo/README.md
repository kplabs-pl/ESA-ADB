This is a fork of https://github.com/ahstat/affiliation-metrics-py

# affiliation-metrics-py

Python 3 implementation of the affiliation metrics and tests for reproducing the experiments described in *Local Evaluation of Time Series Anomaly Detection Algorithms*, accepted in KDD 2022 Research Track: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

### Installation

Type `pip install .` to install the *affiliation*
package. Only the [standard Python library](https://docs.python.org/3/library/index.html) is needed, there is no dependency to external libraries.

### Usage

In a Python session, the following lines give an example for computing 
the affiliation metrics from prediction and ground truth vectors:

```
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events

vector_pred = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
vector_gt   = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1]

events_pred = convert_vector_to_events(vector_pred) # [(4, 5), (8, 9)]
events_gt = convert_vector_to_events(vector_gt)     # [(3, 4), (7, 10)]
Trange = (0, len(vector_pred))

pr_from_events(events_pred, events_gt, Trange)
```

which gives as output:
```
   {'precision': 0.82,
    'recall': 0.84,
    'individual_precision_probabilities': [0.63, 1.0],
    'individual_recall_probabilities': [0.82, 0.87],
    'individual_precision_distances': [0.5, 0.0],
    'individual_recall_distances': [0.5, 0.33]}
```

### Testing and reproducibility

The unit tests can be run by typing:

```
    python -m unittest discover
```

The results from the paper are also tested. 
The specific tests of the results are located at `tests/test_data.py` and tested
against data located in the folder `data/`. 
