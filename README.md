# Naive Supervised Behavior Cloning for OpenAI's MuJoCo Agents in MDP and POMDP setting
```
|-- model.py             (agent in './models/')
|-- train_agent.py       (train agent)
|-- test_agent.py        (agent performance, stores results in './results/') 
```
Install pyTorch and gym requirements (I used an anaconda Python3.6 virtual env).

Neural Network has a validation MSE loss of 0.06 and mean episode rewards of 450 using a current image input grayscaled to (1,96,96) and preprocessed.


 - With/wo belief state
 - With/wo previous actions
 - Different sizes of hidden state in belief module
 - Agents
