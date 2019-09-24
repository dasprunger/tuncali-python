Pythonic Barrier Certificates
============================= 

This work reimplements many components from the paper [Reasoning about Safety of Learning-Enabled Components in 
Autonomous Cyber-physical Systems](https://arxiv.org/abs/1804.03973) mostly in Python. More details about the specifics 
of this implementation can be found below. I'd like to highlight some of the *differences* first.

One major change is to the training algorithm used to find neural network controllers. The controllers from the
original paper are multilayer perceptra with two dense layers, activated with tanh, with 2 inputs and 1 output and a
varying number of neurons in between. Once weights are chosen for this network, the car's dynamics are completely known.
Training the network proceeds by using a target track, numerically solving the differential equation to simulate the 
car on this track, and assessing a penalty based on distance errors, angle errors, and control values. Since the target
track is long, they use an ODE solver to get a reasonable idea of how the car behaves, but using an ODE solver
rules out the use of many white- or grey-box optimization methods. Therefore, the weights of the network are optimized
with CMA-ES, but other black-box optimizers could be used.

As an alternative training method, I propose the use of a large number of short tracks with varying distance and angle
offsets from the ego car. Since these test tracks are very short, we can linearize the differential equation at
each control point and still have a reasonable estimate of how the car behaves. This allows us to sidestep the use of an
ODE solver. In this scheme, we still assign a penalty based on distance errors and angle errors during the (short) run.
However, as a result of the simplified computations, we can use gradient descent to optimize the weights of the
controller network. We also drastically increase the number of test tracks to give the network experience in more
scenarios.

This training method turns out to work reasonably well: we can use it to create networks like in the original paper 
(i.e. with the same architecture) which can still be proven safe with barrier certificates. On my machine, gradient
descent training also seems to be faster than CMA-ES training. 

One peculiarity noticed was that the behavior of the controller depends on the cost function used. When distance 
errors are given a small weight, the controller just learns to travel parallel to the target path (zeroing out the angle
error as quickly as possible). When the distance error weight is large, the resulting controller is underdamped, passing
through the target path as often as possible even though this entails overshooting, since having distance values close 
to 0 radically reduces the penalty. A weight somewhere in the range of 500 times the angle error weight gave a 
reasonable result. I also wonder whether using L1 distances in the phase space might work better for training.

Another change is learning different controller architectures, particularly looking at recurrent/stateful architectures.
I mostly worked on an architecture where the controller has access to the integral of the distance errors. The integral 
of the angle errors is not very different from the distance error itself due to the small angle approximation, so the 
integral of the distance error likely gives more useful information, while minimally extending the dimension of the 
overall system and not complicating the dynamics very much.

It is possible to use this architecture to learn provably safe controllers, and anecdotally it seems these controllers
do better on curved target tracks than purely feedforward controllers. However, we also observed controllers whose phase
portrait had some rather sharp corners, and we weren't able to find barriers using the quadratic form template. Some
ideas on avoiding these problems include changing the cost function to avoid sharp changes in the phase diagrams, or
using a different barrier template.

A few questions to follow up on:
* Can we use stateful controllers with more standard architecture (LSTM, GRU)?
* Can we use the weights of the cost function to help guide the barrier search?
* How do we choose an appropriate cost function?
* Are there better ways to choose the barrier template? 
* Is it possible to use compositional barrier certificates to break down a high-dimensional system into smaller components?
* Can we use the train4safety techniques to keep the stateful controllers in a verifiable region?
* What about control barrier functions?


### Implementation details

This work has a few dependencies: 1) numpy and scipy are standard Python libraries for scientific computing, 2) dReal 4
for satisfiability testing, and 3) torch for neural network primitives and automatic differentiation. The python 
dependencies can be installed with pip and the requirements.txt file. The dReal library has to be built separately 
though.

Some background objects are necessary: `barrier.py` deals with quadratic forms, their Lie derivatives and 
representations, `box_range.py` deals with axis-aligned regions, sampling from them, and creating dReal formulas
expressing membership in them, and `car.py` defines a basic feedforward controller in the style of the paper. We also
keep the definitions of the dynamics of the cars separately in the `vector_fields.py` file; these are actually vector
field factories, which take a control function of a particular format and returns the vector field corresponding to the
closed loop control of the car model combined with the controller.

These controllers can be optimized with `cmaes_car.py`, which optimizes using the CMA-ES scheme from the paper, modified
slightly since I haven't implemented the target path from the paper. `nn_car.py` trains controllers with gradient
descent, as described above. These controllers can be checked for safety with `constraint_generator.py`, which creates
barrier candidates, attempts to check their Lie derivative with respect to the controller is always negative with dReal, 
iteratively refining the candidate while that check fails.

Using these three different math packages (numpy, dreal, torch) in different contexts with their different interfaces is
rather annoying and may lead to implementation errors---we prefer to write, say, the code for the controller once, not
once for each package. So `math_packages.py` provides a common wrapper for these math libraries to make them a bit more
interoperable. `plotting.py` is another utility useful for plotting.

Finally, `rnn_car.py` is the start of the work on stateful controllers, but it is still in a preliminary stage.



