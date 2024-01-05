# Deep Belief Network Example

This DBN implementation is based on the [deep learning book](https://www.deeplearningbook.org/) (page 660). Although the DBN could be utilized as a generative model, in this case, the network was employed to initialize the weights of a Multilayer Perceptron (neural network), which was subsequently fine-tuned for classification.

To achieve this, the DBN was initially separated into multiple Restricted Boltzmann Machines (RBMs). Each RBM was then trained separately, starting with the first one, which modeled the data. Subsequently, the second RBM was trained to model the hidden units of the first one when driven by the data, and so on. The algorithm used for training these RBMs is Contrastive Divergence with k steps (CD-k), as outlined in algorithm 18.2 on page 610.

After training, the DBN could generate new samples by first sampling from the last RBM using Markov chains and then performing ancestral sampling to reach the visible units. Although DBNs are no longer commonly used, even as generative models, they played a pivotal role in the resurgence of deep learning, as mentioned in the book. Additionally, their architecture incorporates both directed and undirected graphical models, which introduces challenges such as the intractable posterior due to the explaining away effect in the directed part and the intractable partition function in the undirected part.


## Getting Started

```bash
git clone https://github.com/oppenheimer3/from_scratch.git
pip install -r requirements.txt
python main.py
```

## options:
``` 
  -h, --help              show this help message and exit.

  --batch BATCH           Batch size for training.

  --dbn_epochs DBN_EPOCHS Number of epochs for training the deep
                          belief network.

  --dbn_lr DBN_LR         Learning rate for training the deep belief
                          network.

  --mlp_epochs MLP_EPOCHS Number of epochs for training the multilayer
                          perceptron.

  --mlp_lr MLP_LR         Learning rate for training the multilayer 
                          perceptron.

  --save                  Save the model after training.
```

