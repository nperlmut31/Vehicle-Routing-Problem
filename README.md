# Vehicle-Routing-Problem

This paper contains neural network based solver for the *capacitated vehicle routing problem* with time-window conatraints, multiple cars, and multiple depots. The model architecture is based on the *transformer* (see [Attention is All You Need](https://arxiv.org/abs/1706.03762), and it is trained using [policy-gradient based reinforcement learning](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) (i.e. the REINFORCE algorithm). 

This project is inspired by the recent paper [Attention, Learn to Solve Routing Problems](https://arxiv.org/abs/1803.08475). Our methods are similar to said paper except that here we apply them to solve more complicated versions of the vehicle routing problem, ones containing multiple cars based at multiple depots and time-window constraints. 

To train the model run the file train.py. The hyperparameters and specifications are contained params.json. To change the hyperparameter, change the items in params.json manualy.
