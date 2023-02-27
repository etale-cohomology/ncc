
----------------------------------------------------------------
# ncc

`ncc` is a **compiler** for neural nets.  
`ncc` reads a representation of a neural net and it produces code for its arithmetic operations.  

`ncc` stands for `neural C compiler`.

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

Output (fwd-pass):

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

Output (bwd-pass, incomplete/incorrect):

```
w0004 =  +DLY_n08*Dn08_w0004 +DLY_n09*Dn09_w0004
w0104 =  +DLY_n08*Dn08_w0104 +DLY_n09*Dn09_w0104
w0005 =  +DLY_n08*Dn08_w0005 +DLY_n09*Dn09_w0005 +DLY_n0a*Dn0a_w0005
w0105 =  +DLY_n08*Dn08_w0105 +DLY_n09*Dn09_w0105 +DLY_n0a*Dn0a_w0105
w0205 =  +DLY_n08*Dn08_w0205 +DLY_n09*Dn09_w0205 +DLY_n0a*Dn0a_w0205
w0106 =  +DLY_n09*Dn09_w0106 +DLY_n0a*Dn0a_w0106 +DLY_n0b*Dn0b_w0106
w0206 =  +DLY_n09*Dn09_w0206 +DLY_n0a*Dn0a_w0206 +DLY_n0b*Dn0b_w0206
w0306 =  +DLY_n09*Dn09_w0306 +DLY_n0a*Dn0a_w0306 +DLY_n0b*Dn0b_w0306
w0207 =  +DLY_n0a*Dn0a_w0207 +DLY_n0b*Dn0b_w0207
w0307 =  +DLY_n0a*Dn0a_w0307 +DLY_n0b*Dn0b_w0307
w0408 =  +DLY_n0c*Dn0c_w0408 +DLY_n0d*Dn0d_w0408
w0508 =  +DLY_n0c*Dn0c_w0508 +DLY_n0d*Dn0d_w0508
w0409 =  +DLY_n0c*Dn0c_w0409 +DLY_n0d*Dn0d_w0409
w0509 =  +DLY_n0c*Dn0c_w0509 +DLY_n0d*Dn0d_w0509
w0609 =  +DLY_n0c*Dn0c_w0609 +DLY_n0d*Dn0d_w0609
w050a =  +DLY_n0c*Dn0c_w050a +DLY_n0d*Dn0d_w050a
w060a =  +DLY_n0c*Dn0c_w060a +DLY_n0d*Dn0d_w060a
w070a =  +DLY_n0c*Dn0c_w070a +DLY_n0d*Dn0d_w070a
w060b =  +DLY_n0c*Dn0c_w060b +DLY_n0d*Dn0d_w060b
w070b =  +DLY_n0c*Dn0c_w070b +DLY_n0d*Dn0d_w070b
w080c = 
w090c = 
w0a0c = 
w0b0c = 
w080d = 
w090d = 
w0a0d = 
w0b0d = 
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

Output (fwd-pass):

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

Another output (fwd-pass):

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

## The fundamental equations

The **fundamental feedforward equation** for neuron `nj` is:

```
nj = fj[SUM[i,Ij, ni*wij]]
```

where index `i` ranges over the set `Ij` of all in-indices into neuron `nj`.

So, the value of each neuron `nj` depends on the value of all the neurons `{ni}` going it `nj`.

This is a recursive equation because each incoming neuron `ni` in turn depends on other neurons, and those depend on others, and so on, until the raw input.

This results in a proliferation of forward calculations, but most of the calculations are repeated, so the complexity of a fwd-pass over the entire net should be `O[N*K+N]`, where `N` is the number of neurons and `K` is the average number of outgoing neuron connections. Equivalently, the complexity should be `O[W+N]`, where `W` is the number of **neuron connections** (aka. "weights"): 1 `mul` for each weight `wij` from neuron `ni` to neuron `nj`, and 1 `fj` evaluation for each neuron `nj`.

Something similar happens in the bwd-pass, in the opposite direction: the loss-weight derivative `DLY_Dwij` for each weight `wij` depends on all the neurons `{nk}` coming **out of** neuron `nj`, and each outgoing neuron `nk` in turn goes into other neurons, and those go into others, and so on, until the final output.

This results in a proliferation of backward calculations, but most of the calculations are repeated: this is the (famous) backpropagation algorithm.

----------------------------------------------------------------
# FCNs (fully connected nets, MLPs)

**Fully connected nets** are an instance of **feedforward-only nets**, like **convolutional nets** or **recurrent nets**.  
They can be described by a Neural Adjacency Matrix.  
FCNs have a with a "layer-wise fully dense" connectivity pattern.  

```
# layer-wise fully dense net (full receptive field, no skip/high connections): 4 input features, 4 hidden neurons, 4 hidden neurons, 2 outputs

## neurons

  layers
  X 1 2 Y
  -------
n|a e i m
e|b f j n
u|c g k
r|d h l
o|
n|
s|
- each letter is a neuron!!!

## input

[a b c d]

## neural adjacency matrix

 abcdefghijklmn
a00001111000000
b00001111000000
c00001111000000
d00001111000000
e00000000111100
f00000000111100
g00000000111100
h00000000111100
i00000000000011
j00000000000011
k00000000000011
l00000000000011
m00000000000000
n00000000000000

## weight block matrix for the entire net

P[0].w:                    [[ ____  ____  ____  ____  __ae  __af  __ag  __ah  ____  ____  ____  ____  ____  ____]  # timestamp 0???  # `ij` means neuron i is connected to neuron j
                            [ ____  ____  ____  ____  __be  __bf  __bg  __bh  ____  ____  ____  ____  ____  ____]
                            [ ____  ____  ____  ____  __ce  __cf  __cg  __ch  ____  ____  ____  ____  ____  ____]
                            [ ____  ____  ____  ____  __de  __df  __dg  __dh  ____  ____  ____  ____  ____  ____]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  __ei  __ej  __ek  __el  ____  ____]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  __fi  __fj  __fk  __fl  ____  ____]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  __gi  __gj  __gk  __gl  ____  ____]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  __hi  __hj  __hk  __hl  ____  ____]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  __im  __in]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  __jm  __jn]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  __km  __kn]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  __lm  __ln]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____]
                            [ ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____]]
------------------------------------------------------------------------
[a b c d e f g h i j k l] | [ ____  ____  ____  ____ +a*ae +a*af +a*ag +a*ah  ____  ____  ____  ____  ____  ____]
                          |   ____  ____  ____  ____ +b*be +b*bf +b*bg +b*bh  ____  ____  ____  ____  ____  ____
                          |   ____  ____  ____  ____ +c*ce +c*cf +c*cg +c*ch  ____  ____  ____  ____  ____  ____
                          |   ____  ____  ____  ____ +d*de +d*df +d*dg +d*dh  ____  ____  ____  ____  ____  ____
                          |   ____  ____  ____  ____  ____  ____  ____  ____ +e*ei +e*ej +e*-k +e*-l  ____  ____
                          |   ____  ____  ____  ____  ____  ____  ____  ____ +f*fi +f*fj +f*fk +f*-l  ____  ____
                          |   ____  ____  ____  ____  ____  ____  ____  ____ +g*gi +g*gj +g*gk +g*gl  ____  ____
                          |   ____  ____  ____  ____  ____  ____  ____  ____ +h*hi +h*hj +h*hk +h*hl  ____  ____
                          |   ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____ +i*im +i*in
                          |   ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____ +j*jm +j*jn
                          |   ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____ +k*km +k*kn
                          |   ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____ +l*lm +l*ln
                          |   ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____
                          |   ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____  ____
                          | -------------------------------------------------------------------------------------
                          |   a     b     c     d     e     f     g     h     i     j     k     l     m     n

# layer-wise fully dense net: 8 input features, 8 neurons, 8 neurons, 2 outputs

## neurons

  layers
  X 1 2 Y
  -------
n|n00 n10 n20 l30
e|n01 n11 n21 l31
u|n02 n12 n22
r|n03 n13 n23
o|n04 n14 n24
n|n05 n15 n25
s|n06 n16 n26
 |n07 n17 n27
- each letter is a neuron!!!

## input

[n00 n01 n02 n03 n04 n05 n06 n07]

## neural adjacency matrix

    nnnnnnnnnnnnnnnnnnnnnnnnnn
  i:00000000111111112222222233
  j:01234567012345670123456701
------------------------------
n00|00000000111111110000000000
n01|00000000111111110000000000
n02|00000000111111110000000000
n03|00000000111111110000000000
n04|00000000111111110000000000
n05|00000000111111110000000000
n06|00000000111111110000000000
n07|00000000111111110000000000
n10|00000000000000001111111100
n11|00000000000000001111111100
n12|00000000000000001111111100
n13|00000000000000001111111100
n14|00000000000000001111111100
n15|00000000000000001111111100
n16|00000000000000001111111100
n17|00000000000000001111111100
n20|00000000000000000000000011
n21|00000000000000000000000011
n22|00000000000000000000000011
n23|00000000000000000000000011
n24|00000000000000000000000011
n25|00000000000000000000000011
n26|00000000000000000000000011
n27|00000000000000000000000011
n30|00000000000000000000000000
n31|00000000000000000000000000
```

----------------------------------------------------------------
# CNNs (convolutional nets)

**Convolutional nets** are an instance of **feedforward-only nets**.

```
# sparsely connected net (conv net w/ 3x3 filters): 9 input features, 2 hidden layers, 3 outputs

## neurons

- input  neurons: n00,n01,n02,n03,n04,n05,n06,n07,n08,n09,n0a,n0b,n0c,n0d,n0e,n0f
- hidden neurons: n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n1a,n1b,n1c,n1d,n1e,n1f,n20,n21,n22,n23,n24,n25,n26,n27,n28,n29,n2a,n2b,n2c,n2d,n2e,n2f,n30,n31,n32,n33,n34,n35,n36,n37,n38,n39,n3a,n3b,n3c,n3d,n3e,n3f
- output neurons: n44,n41,n42

## input

[
	[n00 n01 n02 n03]
	[n04 n05 n06 n07]
	[n08 n09 n0a n0b]
	[n0c n0d n0e n0f]
]

## neural adjacency matrix

    nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
  i:00000000000000001111111111111111222222222222222233333333333333334444444444444444555
  j:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef012
---------------------------------------------------------------------------------------
n00|00000000000000001100110000000000110011000000000000000000000000000000000000000000000
n01|00000000000000001110111000000000111011100000000000000000000000000000000000000000000
n02|00000000000000000111011100000000011101110000000000000000000000000000000000000000000
n03|00000000000000000011001100000000001100110000000000000000000000000000000000000000000
n04|00000000000000001100110011000000110011001100000000000000000000000000000000000000000
n05|00000000000000001110111011100000111011101110000000000000000000000000000000000000000
n06|00000000000000000111011101110000011101110111000000000000000000000000000000000000000
n07|00000000000000000011001100110000001100110011000000000000000000000000000000000000000
n08|00000000000000000000110011001100000011001100110000000000000000000000000000000000000
n09|00000000000000000000111011101110000011101110111000000000000000000000000000000000000
n0a|00000000000000000000011101110111000001110111011100000000000000000000000000000000000
n0b|00000000000000000000001100110011000000110011001100000000000000000000000000000000000
n0c|00000000000000000000000011001100000000001100110000000000000000000000000000000000000
n0d|00000000000000000000000011101110000000001110111000000000000000000000000000000000000
n0e|00000000000000000000000001110111000000000111011100000000000000000000000000000000000
n0f|00000000000000000000000000110011000000000011001100000000000000000000000000000000000
n10|00000000000000000000000000000000000000000000000011001100000000000000000000000000000
n11|00000000000000000000000000000000000000000000000011101110000000000000000000000000000
n12|00000000000000000000000000000000000000000000000001110111000000000000000000000000000
n13|00000000000000000000000000000000000000000000000000110011000000000000000000000000000
n14|00000000000000000000000000000000000000000000000011001100110000000000000000000000000
n15|00000000000000000000000000000000000000000000000011101110111000000000000000000000000
n16|00000000000000000000000000000000000000000000000001110111011100000000000000000000000
n17|00000000000000000000000000000000000000000000000000110011001100000000000000000000000
n18|00000000000000000000000000000000000000000000000000001100110011000000000000000000000
n19|00000000000000000000000000000000000000000000000000001110111011100000000000000000000
n1a|00000000000000000000000000000000000000000000000000000111011101110000000000000000000
n1b|00000000000000000000000000000000000000000000000000000011001100110000000000000000000
n1c|00000000000000000000000000000000000000000000000000000000110011000000000000000000000
n1d|00000000000000000000000000000000000000000000000000000000111011100000000000000000000
n1e|00000000000000000000000000000000000000000000000000000000011101110000000000000000000
n1f|00000000000000000000000000000000000000000000000000000000001100110000000000000000000
n20|00000000000000000000000000000000000000000000000000000000000000001100110000000000000
n21|00000000000000000000000000000000000000000000000000000000000000001110111000000000000
n22|00000000000000000000000000000000000000000000000000000000000000000111011100000000000
n23|00000000000000000000000000000000000000000000000000000000000000000011001100000000000
n24|00000000000000000000000000000000000000000000000000000000000000001100110011000000000
n25|00000000000000000000000000000000000000000000000000000000000000001110111011100000000
n26|00000000000000000000000000000000000000000000000000000000000000000111011101110000000
n27|00000000000000000000000000000000000000000000000000000000000000000011001100110000000
n28|00000000000000000000000000000000000000000000000000000000000000000000110011001100000
n29|00000000000000000000000000000000000000000000000000000000000000000000111011101110000
n2a|00000000000000000000000000000000000000000000000000000000000000000000011101110111000
n2b|00000000000000000000000000000000000000000000000000000000000000000000001100110011000
n2c|00000000000000000000000000000000000000000000000000000000000000000000000011001100000
n2d|00000000000000000000000000000000000000000000000000000000000000000000000011101110000
n2e|00000000000000000000000000000000000000000000000000000000000000000000000001110111000
n2f|00000000000000000000000000000000000000000000000000000000000000000000000000110011000
n30|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n31|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n32|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n33|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n34|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n35|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n36|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n37|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n38|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n39|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n3a|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n3b|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n3c|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n3d|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n3e|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n3f|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n40|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n41|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n42|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n43|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n44|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n45|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n46|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n47|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n48|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n49|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n4a|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n4b|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n4c|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n4d|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n4e|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n4f|00000000000000000000000000000000000000000000000000000000000000000000000000000000111
n50|00000000000000000000000000000000000000000000000000000000000000000000000000000000000
n51|00000000000000000000000000000000000000000000000000000000000000000000000000000000000
n52|00000000000000000000000000000000000000000000000000000000000000000000000000000000000
```

----------------------------------------------------------------
# RNNs (recurrent nets)

**Recurrent nets** are an instance of **feedforward-only nets**.

```
# RNN encoder (output @yt only in the last step)

- ninputs:  2 (input nfeatures, input_size)
- nsteps:   3 (seq_len)
- nhidden:  4
- noutputs: 2 (output nfeatures)
- nlayers:  1

- xlt:                                    input        @ layer l, step t
- Wltx:                                   x weights    @ layer l, step t
- hlt-1:                                  hidden state @ layer l, step t-1
- Wlth:                                   h weights    @ layer l, step t
- Flth:                                   h activ fn   @ layer l, step t

- hlt = Flth[xlt * Wltx + hlt-1 * Wlth]:  hidden state @ layer l, step t
- ylt = Flty[hlt * Wlty]:                 output       @ layer l, step t

## neurons


## input

[ # f0  f1
  [n00,n01],  # step0
  [n02,n03],  # step1
  [n04,n05],  # step2
]

## neural adjacency matrix

    nnnnnnnnnnnnnnnnnnnn
  i:00000000000000001111
  j:0123456789abcdef0123
------------------------
n00|00000011110000000000
n01|00000011110000000000
n02|00000000001111000000
n03|00000000001111000000
n04|00000000000000111100
n05|00000000000000111100
n06|00000000001111000000
n07|00000000001111000000
n08|00000000001111000000
n09|00000000001111000000
n0a|00000000000000111100
n0b|00000000000000111100
n0c|00000000000000111100
n0d|00000000000000111100
n0e|00000000000000000011
n0f|00000000000000000011
n10|00000000000000000011
n11|00000000000000000011
n12|00000000000000000000
n13|00000000000000000000

# RNN encoder (output @yt only in the last step)

- ninputs:  2 (input nfeatures, input_size)
- nsteps:   3 (seq_len)
- nhidden:  4
- noutputs: 2 (output nfeatures)
- nlayers:  2

- xlt:                                    input        @ layer l, step t
- Wltx:                                   x weights    @ layer l, step t
- hlt-1:                                  hidden state @ layer l, step t-1
- Wlth:                                   h weights    @ layer l, step t
- Flth:                                   h activ fn   @ layer l, step t

- hlt = Flth[xlt * Wltx + hlt-1 * Wlth]:  hidden state @ layer l, step t
- ylt = Flty[hlt * Wlty]:                 output       @ layer l, step t

## neurons

## input

[ # f0  f1
  [n00,n01],  # step0
  [n02,n03],  # step1
  [n04,n05],  # step2
]

## neural adjacency matrix
```

# Large Language Models / LLM

- GPT-2,  1.5B: https://github.com/openai/gpt-2
- GPT-Neo 2.7B: https://huggingface.co/EleutherAI/gpt-neo-2.7B
- GPT-J, 6B, 402 billion tokens over 383,500 steps: https://huggingface.co/EleutherAI/gpt-j-6B
- GPT-NeoX
- GPT-3, 175B:
- OPT,   175B
- Bloom, 176B, 1.5TB, 350B tokens: https://huggingface.co/bigscience/bloom
- GLM,   130B: https://github.com/THUDM/GLM-130B

- GPT-Neo 2.7B is about as bad as GPT-3 Ada?

# notes

- https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf
- we require that each sentence ends with a special end-of-sentence symbol “<EOS>”, which enables the model to define a distribution over sequences of all possible lengths
- we used two different LSTMs: one for the input sequence and another for the output sequence, because doing so increases the number model parameters at negligible computational cost and makes it natural to train the LSTM on multiple language pairs simultaneously
- we found that deep LSTMs significantly outperformed shallow LSTMs, so we chose an LSTM with four layers
- we found it extremely valuable to reverse the order of the words of the input sentence
- we parallelized our model using an 8-GPU machine
- each layer of the LSTM was executed on a different GPU and communicated its activations to the next GPU/layer as soon as they were computed (4 GPUs for 4 LSTM layers)
- the remaining 4 GPUs were used to parallelize the softmax
- batch size: 128
- training time: 10 days
- an ensemble of 5 LSTMs with a beam of size 2 is cheaper than of a single LSTM with a beam of size 12

- https://arxiv.org/pdf/1703.03906.pdf
- this suggests that the attention mechanism acts more like a ”weighted skip connection” that optimizes gradient flow than like a ”memory” that allows the encoder to access source states, as is commonly stated in the literature
- it has also been observed (Tu et al., 2017) that very large beam sizes, even with length penalty, perform worse than smaller ones
- Similar to (Tu et al., 2017) we found that very large beams yield worse results and that there is a ”sweet spot” of optimal beam width
- choosing the correct beam width can be crucial to achieving the best results
- Large embeddings with 2048 dimensions achieved the best results, but only by a small margin. Even small embeddings with 128 dimensions seem to have sufficient capacity to capture most of the necessary semantic information
- LSTM Cells consistently outperformed GRU Cells
- Bidirectional encoders with 2 to 4 layers performed best. Deeper encoders were significantly more unstable to train, but show potential if they can be optimized well.
- Deep 4-layer decoders slightly outperformed shallower decoders. Residual connections were necessary to train decoders with 8 layers and dense residual connections offer additional robustness.
- Parameterized additive attention yielded the overall best results

- https://nlp.seas.harvard.edu/2018/04/03/attention.html
- Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions
- With a single attention head, averaging inhibits this

- https://arxiv.org/pdf/1712.09913.pdf
- certain network architecture designs (eg. skip connections) yield loss functions that train easier
- good training parameters (batch size, learning rate, optimizer) produce minimizers that generalize better

- https://weightagnostic.github.io/
- https://arxiv.org/abs/1906.04358
- instead of choosing an architecture and then finding weights, try choosing (random) weights and then finding an architecture
- evaluate a network architecture's performance by how well it performs with RANDOM WEIGHTS!!! |--> replace weight optimization with weight (random) sampling
- try using a single (random) weight for the entire network, of clusters of constant (random) weights (eg. these neural connections have a weight of 0.1, these other neural connections have a weight of 7.3, etc.), or clusters of small variation (ie. these neural connections have a weights with a mean of 7.3 and std of 0.1)
- network pruning: remove neural connections (synapses) where the connection/weight is close to 0 (or negative, if using ReLU?)
- generate an initial population of COMPLEX architectures (or the other way around? go from complex architectures to simple ones, or from simple ones to complex ones?)
- rank/evaluate architectures based on performance AND complexity (lower complexity is better)
- use ensembles of there random-weight nets

- https://arxiv.org/abs/2005.14165
- scaling up Language Models into Large Language Models greatly improves few-shot generalization
- updating the weights of a pre-trained model by training on a supervised dataset
- often thousands to hundreds of thousands of labeled examples are used
- main advantage of fine-tuning: strong performance on many benchmarks
- a disadvance of fine-tuning: the potential for poor generalization out-of-distribution
- fine-turning: update weights
- few-shot:     update no weights, show a few examples
- one-shot:     update no weights, show 1 example
- zero-shot:    update no weights, show 0 examples
- few-shot: an example has a context and a desired completion (eg. an English sentence its French version), and few-shot gives K examples of context and K-1 examples of completion (for the first K-1 examples of context). then the model must provide the K-th completion
- we set K in the range of 10 to 100 as this is how many examples can fit in the model’s context window (nctx = 2048)
- main advantages   of few-shot: much less task-specific data needed. reduced potential to learn a narrow distribution from a large & narrow task-specific dataset
- main disadvantage of few-shot: does much worse than nice fine-tuned models
- gpt-3: similar model & architecture to gpt-2:
  - modified initialization
  - pre-normalization
  - reversible tokenization
  - dense & locally-banded-sparse attertion patterns

- https://aclanthology.org/P19-1334.pdf
- Neural networks are decent at learning statistical patterns in a training set and applying them to test cases drawn from the same distribution as the training examples
- Statistical learners such as neural networks closely track the statistical regularities in their training sets. This process makes them vulnerable to adopting heuristics that are valid for frequent cases but fail on less frequent ones

- https://arxiv.org/abs/1904.09751
- The top quality content from gpt-2 relies on randomness in the decoding method: top-k sampling [1904.09751]
- top-k sampling: top k toks with highest proba
- top-p sampling: top   toks with proba at least p
- beam search leads to very bad text

- https://arxiv.org/abs/1610.06258
- https://arxiv.org/abs/1503.05671
- https://arxiv.org/abs/1603.07285
- https://arxiv.org/abs/1810.03798
- https://arxiv.org/abs/1709.06080
- https://arxiv.org/abs/2007.03347
- https://arxiv.org/abs/1611.01578
- https://arxiv.org/abs/2206.10789
- https://arxiv.org/abs/2006.04439
- https://aclanthology.org/2022.acl-long.26.pdf
- https://arxiv.org/abs/2208.12242
- https://arxiv.org/abs/2206.10789
- https://arxiv.org/abs/2105.01601
- https://arxiv.org/abs/1710.03740
- https://arxiv.org/abs/1807.03748
- https://arxiv.org/abs/2110.02861
- https://royalsocietypublishing.org/doi/pdf/10.1098/rspb.2012.2863
- https://arxiv.org/abs/1906.04358
- http://proceedings.mlr.press/v49/eldan16.pdf
- https://proceedings.neurips.cc/paper/2017/file/32cbf687880eb1674a07bf717761dd3a-Paper.pdf
- https://arxiv.org/abs/1702.08580
- https://arxiv.org/abs/1702.05777
- https://arxiv.org/abs/2005.14165
- https://arxiv.org/abs/1907.11692
- https://pjreddie.com/media/files/papers/xnor.pdf
- https://arxiv.org/abs/1310.6343
- https://arxiv.org/abs/2205.14135
- https://openai.com/blog/requests-for-research-2/
- https://ycombinator.com/rfs/
- https://github.com/karpathy/minGPT/
- https://github.com/karpathy/nanoGPT
- https://arxiv.org/abs/1904.09751
- https://thomwolf.io/data/Thom_wolf_reading_list.txt
- https://colab.research.google.com/drive/1yeLM1LoaEqTAS6D_Op9_L2pLA06uUGW1#scrollTo=pfPQuW-2u-Ps
- https://arxiv.org/abs/1612.03144
- https://arxiv.org/abs/1710.09412v2
- https://arxiv.org/abs/2012.14905
- https://arxiv.org/abs/1810.03798
- https://arxiv.org/abs/1702.07800
- https://arxiv.org/abs/2105.01601
- https://arxiv.org/abs/2101.11986
- https://arxiv.org/abs/2210.15424
- https://arxiv.org/abs/2208.12242
- https://arxiv.org/abs/2205.06175
- https://arxiv.org/abs/2201.08239
- https://arxiv.org/abs/2202.09778
- https://arxiv.org/abs/1903.03129
- https://arxiv.org/abs/2010.11929v2
- https://arxiv.org/abs/1706.05137
- https://arxiv.org/abs/2109.03910v4
- https://arxiv.org/abs/2105.07581
- https://arxiv.org/abs/1709.05011v10
- http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
- https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html
- https://victorzhou.com/blog/intro-to-cnns-part-2/
- https://kdnuggets.com/2022/10/abcs-nlp-a-to-z.html
- https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
- https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Dynamical_systems/dynamical_systems_neural_odes.html
- http://columbia.edu/~pg2113/index_files/Gorroochurn-Some%20Laws.pdf

- https://github.com/HazyResearch/flash-attention
