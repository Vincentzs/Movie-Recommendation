# Deep Reinforcement Learning based Recommender System in Tensorflow
The implemetation of Deep Reinforcement Learning based Recommender System from the paper [Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling](https://arxiv.org/abs/1810.12027) by Liu et al. Build recommender system with [DDPG](https://arxiv.org/abs/1509.02971) algorithm. Add state representation module to produce trainable state for RL algorithm from data. ***This is not the official implementation of the paper***.

# Dataset
[MovieLens 1M Datset](https://grouplens.org/datasets/movielens/1m/)

```
unzip ./ml-1m.zip
```

# Procedure
- Trying to improve performance of RL based recommender system. The report contains the result of Using the actor network with embedding layer, reducing overestimated Q value, using several pretrained embedding and applying [PER](https://arxiv.org/abs/1511.05952).

- Making new embedding files. Previous one contains the information for entire timelines which can mislead model.

- Updating Train and Evaluation part. Currently, I didn't follow the exact same training and evaluation procedure in the paper.



# Result

### Please check here - [Experiment Report (Korean)](https://www.notion.so/DRR-8e910fc598d242968bd371b27ac20e01)

<br>

![image](https://user-images.githubusercontent.com/30210944/109442330-40b37180-7a7b-11eb-8303-d45a8083dbc7.png)

- for evalutation data
    - precision@5 : 0.479, ndcg@5 : 0.471
    - precision@10 : 0.444, ndcg@10 : 0.429

# Usage
### Training
- The saved model of actor and critic are generated after the training is done.
```
python train.py
```
### Evalutation
- Make sure there exist the saved models in the right directory
- Run jupyter notebook and check "evaluation.ipynb"

# requirements
```
tensorflow==2.5.0
scikit-learn==0.23.2
matplotlib==3.3.3
```

# reference

https://github.com/LeejwUniverse/RL_Rainbow_Pytorch

https://github.com/kyunghoon-jung/MacaronRL

https://github.com/pasus/Reinforcement-Learning-Book
