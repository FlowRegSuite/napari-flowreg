## 🚧 Under Development

This project is still in an **alpha stage**. Expect rapid changes, incomplete features, and possible breaking updates between releases. 

- The API may evolve as we stabilize core functionality.  
- Documentation and examples are incomplete.  
- Feedback and bug reports are especially valuable at this stage.

# <img src="img/flowreglogo.png" alt="FlowReg logo" height="64"> napari-FlowReg

This repository contains the napari wrapper for the Flow-Registration toolbox, which is a toolbox for the compensation and stabilization of multichannel microscopy videos. 
The publication for this toolbox can be found [here](https://doi.org/10.1002/jbio.202100330) and the project website with video results [here](https://www.snnu.uni-saarland.de/flow-registration/).

**Related projects**
- PyFlowReg: https://github.com/FlowRegSuite/pyflowreg
- Original Flow-Registration repo: https://github.com/FlowRegSuite/flow_registration
- ImageJ/Fiji plugin: https://github.com/FlowRegSuite/flow_registration_IJ


![Fig1](img/bg.jpg)


## Installation via pip and conda

To install the plugin via conda, you can create a new environment and install `napari` along with the plugin:

    conda create -n flowreg -c conda-forge python=3.12.0 napari

You can then install `napari-flowreg` via [pip]:

    pip install napari[all] napari-flowreg

or from the directly from the GitHub repository:

    pip install git+https://github.com/flowregsuite/napari-flowreg.git


## Dataset

The dataset which we used for our evaluations is available as [2-Photon Movies with Motion Artifacts](https://drive.google.com/drive/folders/1fPdzQo5SiA-62k4eHF0ZaKJDt1vmTVed?usp=sharing).

## Citation

Details on the original method and video results can be found [here](https://www.snnu.uni-saarland.de/flow-registration/).

If you use parts of this code or the plugin for your work, please cite

> “Pyflowreg,” (in preparation), 2025.

and 

> P. Flotho, S. Nomura, B. Kuhn and D. J. Strauss, “Software for Non-Parametric Image Registration of 2-Photon Imaging Data,” J Biophotonics, 2022. [doi:https://doi.org/10.1002/jbio.202100330](https://doi.org/10.1002/jbio.202100330)

BibTeX entry
```
@article{flotea2022a,
    author = {Flotho, P. and Nomura, S. and Kuhn, B. and Strauss, D. J.},
    title = {Software for Non-Parametric Image Registration of 2-Photon Imaging Data},
    year = {2022},
  journal = {J Biophotonics},
  doi = {https://doi.org/10.1002/jbio.202100330}
}
```