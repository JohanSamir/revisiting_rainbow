# Revisiting Rainbow: Promoting more insightful and inclusive deep reinforcement learning research

In this work we argue that, despite the community’s emphasis on large-scale environments, the traditional small-scale environments can
still yield valuable scientific insights and can help reduce the barriers to entry for underprivileged communities. To substantiate our claims, we empirically revisit the paper which introduced the Rainbow algorithm [Hessel et al., 2018][fortunato] and present some new insights into the algorithms used by Rainbow.

Our rainbow agent implements three addittional components to the rainbow agent proposed by Dopamine. ([Pablo Samuel Castro et al., 2018][castro])

* Noisy nets ([Meire Fortunato et al., 2018][fortunato])
* Dueling networks  ([Hado van Hasselt et al., 2016][wang])
* Double Q-learning ([Ziyu Wang et al., 2016][hasselt])
* Munchausen Reinforcement Learning ([Nino Vieillard et al., 2020][Vieillard])

if you are interested to know more about Revisiting Rainbow, considering check the following resources:

* **Paper:** [arxiv.org/abs/2011.14826][arXiv_rev] 
* **Blog:** [https://psc-g.github.io/posts/...][blog]
* **Deep RL Workshop talk, NeurIPS 2020:** [https://slideslive.com/38941329/...][video]


## Quick Start
To use the algorithms proposed in the Revisiting Rainbow paper, you need python3 installed, make sure pip is also up to date.  If you want to run the MinAtar experiments you should install it. To install MinAtar, please check the following paper ([Young et al., 2019][young]) and repositore ([github][young_repo]): 

1. Clone the repo: 
```bash
https://github.com/JohanSamir/revisiting_rainbow
```
If you prefer running the algorithms in a virtualenv, you can do the following before step 2:

```bash
python3 -m venv venv
source venv/bin/activate
# Upgrade Pip
pip install --upgrade pip
```

2.  Finally setup the environment and install Revisiting Rainbow's dependencies
```bash
pip install -U pip
pip install -r revisiting_rainbow/requirements.txt
```

## Running tests

Check the following colab file [`revisiting_rainbow/test_main.ipynb`](https://github.com/revisiting_rainbow/test_main.ipynb) to run the basic DQN agent.

## References

[Hado van Hasselt, Arthur Guez, and David Silver. *Deep reinforcement learning with double q-learning*. 
In Proceedings of the Thirthieth AAAI Conference On Artificial Intelligence (AAAI), 2016.][hasselt]

[Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan
Horgan, Bilal Piot, Mohammad Azar, and David Silver. *Rainbow: Combining Improvements in Deep Reinforcement learning*.
In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.][Hessel]

[Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alexander
Graves, Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, and
Shane Legg. *Noisy networks for exploration*. In Proceedings of the International Conference on
Representation Learning (ICLR 2018), Vancouver (Canada), 2018.][fortunato]

[Pablo Samuel Castro, Subhodeep Moitra, Carles Gelada, Saurabh Kumar, and Marc G. Bellemare.
*Dopamine: A Research Framework for Deep Reinforcement Learning*, 2018.][castro]

[Kenny Young and Tian Tian. *Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments*, 2019.][young]

[Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Hasselt, Marc Lanctot, and Nando Freitas. *Dueling network architectures for deep reinforcement learning*. In Proceedings of the 33rd International
Conference on Machine Learning, volume 48, pages 1995–2003, 2016.][wang]

[Vieillard, N., Pietquin, O., and Geist, M. Munchausen Reinforcement Learning. In Advances in Neural Information Processing Systems (NeurIPS), 2020.][Vieillard]

[fortunato]: https://arxiv.org/abs/1706.10295
[hasselt]: https://arxiv.org/abs/1509.06461
[wang]: https://arxiv.org/abs/1511.06581
[castro]: https://arxiv.org/abs/1812.06110
[Hessel]: https://arxiv.org/abs/1710.02298
[young]: https://arxiv.org/abs/1903.03176
[Vieillard]: https://arxiv.org/abs/2007.14430
[young_repo]: https://github.com/kenjyoung/MinAtar
[arXiv_rev]: https://arxiv.org/abs/2011.14826
[blog]: https://psc-g.github.io/posts/research/rl/revisiting_rainbow/
[video]: https://slideslive.com/38941329/revisiting-rainbow-promoting-more-insightful-and-inclusive-deep-reinforcement-learning-research

## Giving credit
If you use Revisiting Rainbow in your research please cite the following:

Johan S Obando-Ceron, & Pablo Samuel Castro (2020). Revisiting Rainbow: Promoting more insightful and inclusive deep reinforcement learning research. Proceedings of the 38th International Conference on Machine Learning, ICML 2021. [*arXiv preprint:* ][arXiv_rev]

In BibTeX format:

```
@inproceedings{obando2020revisiting,
  title={Revisiting Rainbow: Promoting more insightful and inclusive deep reinforcement learning research},
  author={Obando-Ceron, Johan S and Castro, Pablo Samuel},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  year = {2021},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
}
```
