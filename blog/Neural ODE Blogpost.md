 # Neural ODEs


I recently learned about neural ODEs and I found some things a bit confusing at first so I thought I might explain them from my perspective to possibly help others to better or easier understand the topic. That being said, there are other good resources to this material, I'll link to them later in the post.

### Ok so what *are* neural ODEs?
Most importantly, they are *not* a new network architecture by themselves. I like to think of them as a framework *around* an existing neural network, changing the interpretation of that network and they way it is used.


I'll assume some familiarity with multivariate calculus for this post.

Let's say you have some system, where you don't quite know how it evolves over time, but you can periodically measure it's state. You decide that this system likely follows an ordinary differential equation (ODE). If we denote the state of the system at time $t$ with $y(t)$, then an ODE of order $n$ is an equation of the form $y^{(n)} = F(t, y^{(n-1)}, \ldots, y)$, i.e. the $n-th$ derivative of $y$ behaves according to some function $F$ of the lower order derivatives, and potentially explicitly of the time $t$.
 
For the remainder of this post we will only look at first-order ODEs, i.e. equations of the form $\dot y(t) = f(y(t), t)$ for some function $f$. This is not as restrictive as it looks since we can introduce auxiliary variables $z(t) = \dot y(t)$ to create a system of first-order ODEs from a second order ODE, and repeat the process for higher order ODEs [citation needed].
 
 
For a neural ODE you need two essential things:

* A neural network. This network should take as input a value $y(t)$ and a time $t$ and put out its estimate of $f(y(t), t)$. Almost any commonly used neural network architecture is fair game. Some forms of attention mechanisms have problems around the uniqueness of the found solutions because they are not Lipschitz, but except for those, go nuts. 
*  A (blackbox) ODE solver. This solver should take as input our function $f$ from above, the initial value $y(t_0)$ and an end time $t^\star$. It then calculates (an approximation of) $y(t^\star)$ based on the given dynamics model $f$. I won't go into any detail of how ODE solvers work.
Okay, now that we have those two building blocks, we can build our neural ODE model! 

### How the Neural ODE model computes its prediction:
* The input to the model is a tuple of the initial state $y(t_0)$, the initial time $t_0$ and the end time $t^\star$.
* At step $k$ in the training process the neural network function is given as $f_k(y, t)$. 
* For the given initial state, initial time and dynamics model (given by the neural network), the ODE solver now computes $y(t^\star)$, i.e. the state of the system at time $t^\star$ if started in $y(t_0)$ at time $t_0$ and the system evolves under $f_k(y, t)$.
* This predicted $y(t^\star)$ can now be scored against our ground truth in the training set, e.g. via the MSE loss.

Usually this is the point where we call the backward pass function and let the autodiff library of our choice do its magic. But... how the *hell* do we compute the gradients of the loss in this case? We have a blackbox ODE which calls our neural network many many times over the course of a single forward pass and its outputs at every step depend on all the previous outputs. Yes, we could do BPTT, but that would be very expensive and potentially suffer from similar issues as long-run dependencies in RNNs.

Luckily, there is a solution to more efficiently compute the gradient, using something called the adjoint method.
 
