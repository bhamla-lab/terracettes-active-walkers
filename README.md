# Moving mountains: grazing agents drive terracette formation on steep hillslopes

## Overview

During a field trip to the Swiss Alps, we observed dramatic, step-like landforms on steep hillslopes, which we later identified as terracettes (see attached photo). Upon further investigation, we discovered that the origin of terracettes has been debated among scientists for over a century. One hypothesis attributes their formation to geophysical mass-wasting processes; the other, more controversial, suggests they arise from the repeated trampling of grazing animals. A long-standing objection to the latter hypothesis is that animal movement is too irregular to produce such ordered patterns. Yet no prior model has successfully linked local foraging behavior to the emergence of landscape-scale structure.

To address this, we developed a computational model in which grazing animals are represented as active random agents moving across an erodible slope. Each agent evaluates an energetic tradeoff between locomotion effort and forage reward, with each footstep compacting the soil and depleting local vegetation. These landscape modifications influence future movement through stigmergic interaction. As agents are biased to favor easier paths, their traffic concentrates into cross-slope trails. Over time, these trails self-organize into periodic bands that closely resemble terracettes in both form and spacing.



Contents:

```
terracettes-active-walkers/
├─ README.md
├─ data/
├─ src/                       # your reusable code lives here
├─ notebooks/
```

<p align="center">
  <img src="/animated_erosion.gif" alt="Demo" width="60%" height="60%">
</p>


## Publications

This work is available on **arXiv** (preprint).

If you use this work in an academic context, please cite the following publication:

> **B. Seleb, A. Chatterjee, and S. Bhamla**  
> *"Moving mountains: grazing agents drive terracette formation on steep hillslopes"*,  
> [arXiv:2504.17496](https://arxiv.org/abs/2504.17496), 2025.

```bibtex
@misc{seleb2025movingmountains,
      title={Moving mountains: grazing agents drive terracette formation on steep hillslopes}, 
      author={Benjamin Seleb and Atanu Chatterjee and Saad Bhamla},
      year={2025},
      eprint={2504.17496},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      doi={10.48550/arXiv.2504.17496}
}
```
