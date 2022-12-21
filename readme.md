# ncc

`ncc` is a **compiler** for neural nets.  
`ncc` reads a representation of a neural net and it produces code for its arithmetic operations.  

**Example (Neural Adjacency List: in-indices).**

Input:

```
% after the N line, every line has the form: j fj Ij, where j is the neuron index, fj is the activation function for neuron nj, Ij is the set of in-indices for neuron nj
N 0e
00
01
02
03
04 03 00,01
05 03 00,01,02
06 03 01,02,03
07 03 02,03
08 03 04,05
09 03 04,05,06
0a 03 05,06,07
0b 03 06,07
0c 03 08,09,0a,0b
0d 03 08,09,0a,0b
```

Output:

```
n04 = f04( +n00*w0004 +n01*w0104)
n05 = f05( +n00*w0005 +n01*w0105 +n02*w0205)
n06 = f06( +n01*w0106 +n02*w0206 +n03*w0306)
n07 = f07( +n02*w0207 +n03*w0307)
n08 = f08( +n04*w0408 +n05*w0508)
n09 = f09( +n04*w0409 +n05*w0509 +n06*w0609)
n0a = f0a( +n05*w050a +n06*w060a +n07*w070a)
n0b = f0b( +n06*w060b +n07*w070b)
n0c = f0c( +n08*w080c +n09*w090c +n0a*w0a0c +n0b*w0b0c)
n0d = f0d( +n08*w080d +n09*w090d +n0a*w0a0d +n0b*w0b0d)
```

**Example (Neural Adjacency Matrix: if entry ij is 1, then neuron i goes into neuron j).**

Input:

```
00001100000000
00001110000000
00000111000000
00000011000000
00000000110000
00000000111000
00000000011100
00000000001100
00000000000011
00000000000011
00000000000011
00000000000011
00000000000000
00000000000000
```

Output:

```
n00 = f00()
n01 = f01()
n02 = f02()
n03 = f03()
n04 = f04( +n00*w0004 +n01*w0104)
n05 = f05( +n00*w0005 +n01*w0105 +n02*w0205)
n06 = f06( +n01*w0106 +n02*w0206 +n03*w0306)
n07 = f07( +n02*w0207 +n03*w0307)
n08 = f08( +n04*w0408 +n05*w0508)
n09 = f09( +n04*w0409 +n05*w0509 +n06*w0609)
n0a = f0a( +n05*w050a +n06*w060a +n07*w070a)
n0b = f0b( +n06*w060b +n07*w070b)
n0c = f0c( +n08*w080c +n09*w090c +n0a*w0a0c +n0b*w0b0c)
n0d = f0d( +n08*w080d +n09*w090d +n0a*w0a0d +n0b*w0b0d)
```

Another output:

```
14 1  196 56
        n00        n01        n02        n03        n04        n05        n06        n07        n08        n09        n0a        n0b        n0c        n0d
n00 __________ __________ __________ __________ +n00*w0004 +n00*w0005 __________ __________ __________ __________ __________ __________ __________ __________
n01 __________ __________ __________ __________ +n01*w0104 +n01*w0105 +n01*w0106 __________ __________ __________ __________ __________ __________ __________
n02 __________ __________ __________ __________ __________ +n02*w0205 +n02*w0206 +n02*w0207 __________ __________ __________ __________ __________ __________
n03 __________ __________ __________ __________ __________ __________ +n03*w0306 +n03*w0307 __________ __________ __________ __________ __________ __________
n04 __________ __________ __________ __________ __________ __________ __________ __________ +n04*w0408 +n04*w0409 __________ __________ __________ __________
n05 __________ __________ __________ __________ __________ __________ __________ __________ +n05*w0508 +n05*w0509 +n05*w050a __________ __________ __________
n06 __________ __________ __________ __________ __________ __________ __________ __________ __________ +n06*w0609 +n06*w060a +n06*w060b __________ __________
n07 __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ +n07*w070a +n07*w070b __________ __________
n08 __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ +n08*w080c +n08*w080d
n09 __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ +n09*w090c +n09*w090d
n0a __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ +n0a*w0a0c +n0a*w0a0d
n0b __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ +n0b*w0b0c +n0b*w0b0d
n0c __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________
n0d __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________ __________
```


`ncc` is only intended to work with feedforward-only neural nets, ie. nets that admit only feedforward connections (examples: fully connected nets, convolutional nets, recurrent nets), as opposed to feedforward/feedback neural nets, like Deep Boltzman Machines.

Currently only the forward pass works.

# What is a neural net

A **neural net** is a **directed graph**, where the **vertices** are the **neurons** and the **edges** are the **neuron connections**.  
The **neuron connections** are the (famous) set of **weights** (each weight `wij` is a connection from neuron `ni` to neuron `nj`).  

Since a neural net is a graph, it can be represented in multiple ways, 3 of which are:

0. an adjacency matrix: if entry ij is 1, then neuron `ni` goes into neuron `nj`)
1. an adjacency list: a map from each vertex `v` to a list of vertices (the vertices that `v` is connected to, or, equivalently, the vertices that are connected to `v`)
2. a list of edges
