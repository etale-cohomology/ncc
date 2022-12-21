# ncc

`ncc` is a **compiler** for neural nets.  
`ncc` reads a representation of a neural net and it produces code for its arithmetic operations.  

**Example.**

Input:

```
```

Output:
```
hi
```

`ncc` is only intended to work with feedforward-only neural nets, ie. nets that admit only feedforward connections (examples: fully connected nets, convolutional nets, recurrent nets), as opposed to feedforward/feedback neural nets, like Deep Boltzman Machines.

Currently only the forward pass works.

# What is a neural net

A **neural net** is a **directed graph**, where the **vertices** are the **neurons** and the **edges** are the **neuron connections**.  
The **neuron connections** are the (famous) set of **weights**.  

Since a neural net is a graph, it can be represented in multiple ways, 3 of which are:

0. an adjacency matrix
1. a map from each vertex V to a list of vertices (the vertices that V is connected to, or, equivalently, the vertices that are connected to V)
2. a list of edges
