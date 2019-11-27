# LocIT

This repository contains the online supplement of the 2020 AAAI paper **Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection**. It contains: *the appendix*, *the code for the experiments*, and *the benchmark data*.

The paper authors are from the [DTAI group](https://dtai.cs.kuleuven.be/) of the [KU Leuven](https://kuleuven.be/):

- [Vincent Vercruyssen](https://people.cs.kuleuven.be/~vincent.vercruyssen/)
- [Wannes Meert](https://people.cs.kuleuven.be/~wannes.meert/)
- [Jesse Davis](https://people.cs.kuleuven.be/~jesse.davis/)


## Abstract

> *Anomaly detection attempts to identify instances that deviate from expected behavior. Constructing performant anomaly detectors on real-world problems often requires some labeled data, which can be difficult and costly to obtain. However, often one considers multiple, related anomaly detection tasks. Therefore, it may be possible to transfer labeled instances from a related anomaly detection task to the problem at hand. This paper proposes a novel transfer learning algorithm for anomaly detection that selects and transfers relevant labeled instances from a source anomaly detection task to a target one. Then, it classifies target instances using a novel semi-supervised nearest-neighbors technique that considers both unlabeled target and transferred, labeled source instances. The algorithm outperforms a multitude of state-of-the-art transfer learning methods and unsupervised anomaly detection methods on a large benchmark. Furthermore, it outperforms its rivals on a real-world task of detecting anomalous water usage in retail stores.*

In short, the paper tackles the following task:

```java
GIVEN:  a (partially) labeled source dataset Ds and
        an unlabeled target dataset Dt from the same feature space;

DO:     assign an anomaly score to each instance in Dt
        using both Dt and a subset of Ds.
```

The **appendix to the paper** (and the full conference paper) can either be accessed in `LocIT/paper/` or via the [webpage](https://people.cs.kuleuven.be/~vincent.vercruyssen/).


## Code and data

The Python code for the full **LocIT** algorithm along with the baselines used is in the folder: `LocIT/models/`

The benchmark datasets and the scripts to construct them are in the folder: `LocIT/data/`


#### *DISCLAIMER*:

The paper introduces **LocIT** as the combination of a *transfer learning* subroutine and a *semi-supervised anomaly detection* subroutine. We recommend to use these subroutines separately in your own work. Therfore, both are contained in their own *pip-installable* packages!
The transfer learning subroutine is also called **LocIT** and is implemented in the [*transfertools*](https://github.com/Vincent-Vercruyssen/transfertools) package.
The semi-supervised anomaly detection subroutine is called **SSkNNO** and is implemented in the [*anomatools*](https://github.com/Vincent-Vercruyssen/anomatools) package.
```bash
pip install transfertools
pip install anomatools
```
Once installed, the models can be used as follows:
```python
from transfertools.models import LocIT
from anomatools.models import SSkNNO
```


## Contact

Feel free to ask questions: [vincent.vercruyssen@kuleuven.be](mailto:vincent.vercruyssen@kuleuven.be)


## Citing the AAAI paper

```
@inproceedings{vercruyssen2020transfer,
    title       = {Transfer Learning for Anomaly Detection through Localized and Unsupervised Instance Selection},
    author      = {Vercruyssen, Vincent and Meert, Wannes and Davis Jesse},
    booktitle   = {Thirty-Fourth AAAI Conference on Artificial Intelligence},
    year        = {2020}
}
```