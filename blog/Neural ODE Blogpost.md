 # Neural ODEs


I recently learned about neural ODEs and I found some things a bit confusing at first so I thought I might explain them from my perspective to possibly help others to better or easier understand the topic. That being said, there are other good resources to this material, I'll link to them later in the post.

I'll assume some familiarity with multivariate calculus for this post.

## Ordinary Differential Equations

Let's say you have some system $y(t)$ of which you don't quite know how it evolves over time. For example this system could be a pendulum, and since you didn't pay attention in your physics class you forgot how it behaves.

Since you remember the lecturer saying something about differential equations you decide that the pendulum likely follows an ordinary differential equation (ODE). This is not unreasonable since it seems to be true of the physical world in many aspects. So you review what you know about ODEs...

An ODE of order $n$ is an equation of the form 
$$y^{(n)} = f(y^{(n-1)}, \ldots, y, t),$$
i.e. the $n$-th derivative of $y$ behaves according to some function $f$ of the lower order derivatives, and potentially explicitly of the time $t$.
 
For the remainder of this post we will only look at *first-order* ODEs, i.e. equations of the form 
$$\dot y(t) = f(y(t), t)$$
for some function $f$ ($\dot y(t)$ is a different way of writing $\frac{\partial y}{\partial t}(t)$ ). 

This is not as restrictive as it may seem since we can introduce auxiliary variables $z(t) = \dot y(t)$ to create a system of first-order ODEs from a second order ODE, and repeat the process for higher order ODEs^[citation\ needed]^.

In the example of the pendulum, you would for example look at the angle $y(t) := \phi(t)$ that the pendulum has to the vertical line at time $t$. In that case, $f(\phi(t), t)$ describes how the angle of the pendulum changes over time. 

If you knew the function $f$ you could use known techniques to *solve* the ODE. Solving means that you can compute the exact coordinates $\phi(t^\star)$ at some point in time $t^\star$, given the starting point $\phi(t_0)$ of the pendulum at some initial time $t_0$. However, you do not know the exact dynamics $f$. But what you do have are some measurements of the angle at some points in time after the pendulum was let go at $t_0$. 

How can you predict the dynamics of the system, given this data? 

Enter Neural ODEs!



## Ok so what *are* neural ODEs?


For a neural ODE you need two essential things:

1. A neural network. This network should take as input a value $y(t)$ and a time $t$ and put out its estimate of $f(y(t), t)$. Almost any commonly used neural network architecture is fair game. Some forms of attention mechanisms have problems around the uniqueness of the found solutions because they are not Lipschitz, but except for those, go nuts. 
    * That means the neural network is *interpreted* as modeling the dynamics of the system in question.
2.  A (blackbox) ODE solver. This solver should take as input our function $f$ from above, the initial value $y(t_0)$ and an end time $t^\star$. It then calculates (an approximation of) $y(t^\star)$ based on the given dynamics model $f$. I won't go into any detail of how ODE solvers work.

Importantly, Neural ODEs are *not* a new network architecture by themselves. I like to think of them more as a framework *around* an existing neural network, changing the *interpretation* of that network and they way it is used.

Okay, now that we have those two building blocks, we can build our neural ODE model! 



## How the Neural ODE model computes its prediction:
* The input to the model is a tuple of the initial state $y(t_0)$, the initial time $t_0$ and the end time $t^\star$.
* At step $k$ in the training process, the neural network function is given as $f_k:(y, t)\mapsto \dot y(t)$. 
* For the given initial state, initial time and dynamics model, the ODE solver now computes $y(t^\star)$, i.e. the state of the system at time $t^\star$ if started in $y(t_0)$ at time $t_0$ and the system evolves under $f_k(y, t)$.
* This predicted $y(t^\star)$ can now be scored against our ground truth in the training set, e.g. via the MSE loss.




## Backprop
Usually this is the point where we call the backward pass function and let the autodiff library of our choice do its magic. But... how the *hell* do we compute the gradients of the loss in this case? We have a blackbox ODE which calls our neural network many many times over the course of a single forward pass and its outputs at every step depend on all the previous outputs. Standard backprop would be very costly in this case!

Luckily, there is a solution to more efficiently compute the gradient, using something called the adjoint method.
 
For now, I will not go into that in more detail, but there are good resources on this, not least of which is the original Neural ODE paper.


## Use Cases
So far we always treated the network as modeling the dynamics of a real, physical system, and that is certainly one application of the idea. However, we can also conceptualize the computation of an arbitrary network as a dynamical system: 

* initial state: input to the network (e.g. an image) 
* start time: 0 (chosen arbitrarily)
* final state: prediction (e.g. some score between 0 and 1).
* final time: 1 (arbitrary)

Then, the system's state start at the input and then evolves under the network-given dynamics towards the output. 

Framing the standard supervised classification problem this way means we can extend our neural ODE model to that whole class of problems as well! Similarly with generative models, such as normalizing flows.

