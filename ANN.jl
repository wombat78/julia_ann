#########################################################################
#                                                                       #
# ANN - support functions for neural network training                   #
# Author: (c) Boon Pang Lim                                             #
#   Institute for Infocomm Research, Human Language Technology Dept     #
#   Agency for Science Technology and Research Singapore                #
#                                                                       #
# Date: Wed Jan  7 12:21:40 SGT 2015                                    #
#                                                                       #
# Included functions:                                                   #
#   Sigmoid calculation                                                 #
#   Derivative calculations                                             #
#       - J with L2 Regularization                                      #
#   Simple MLP training and computation                                 #
#       - back propagation                                              #
#       - stochastic gradient descent                                   #
#                                                                       #
# References: http://ufldl.stanford.edu/wiki/index.php/Neural_Networks  #
#                                                                       #
#########################################################################

import Winston;     # required for the plotting functions

module ANN 

using Base.Test     # require support for unit testing

ALPHA=1.0       # learning rate
LAMBDA=1e-8     # regularization - this needs to be small so that the 
                # objF has a smooth manifold
EPSILON=1e-4    # epsilon to use when using numerical differentiation

####
## helper functions (math)
####

# Sigmoid family --> restricts inputs to within 0 and 1
function Sigmoid(x)
   return 1/(1+exp(-x))
end

function dSigmoid(x)
    return Sigmoid(x)*(1-Sigmoid(x))
end

# To implement (Tanh)

# derive - numerically calculates derivatives of functions 
# uses shortcuts if they are known.
function derive(func,delta=EPSILON)
    # provide optimized code for some special cases
    if func == Sigmoid 
        f=dSigmoid 
    else # general case for unknown functions
        f = (x) -> (func(x+delta)-func(x-delta))/(2*delta)
    end
    return f
end

function derivative(func,x,delta=1e-10)
    return (func(x+delta)-func(x-delta))/(2*delta)
end


# expand_labels takes a list of labels (int), and expands it to vector format
#   * useful for training classifiers
function expand_labels(labels,last_label=0)
    if last_label == 0
        last_label=max(last_label,maximum(labels))
    end
    targets=zeros(last_label,length(labels))
    for i in 1:length(labels)
        targets[labels[i],i] = 1 
    end
    return targets
end

####
# Implementaion of basic MLP using an explicit data type
####

# Data structure to hold an mlp type -- new functions are created 
# these are uniform networks with only one type neuron with same activation
# accessed as mlp_FUNC
type MLP
    num_layers  :: Int
    weights     :: Array{ Array {Float64,2 } }
    bias        :: Array{ Array {Float64,2 } }
    activationF :: Function
end

## creation, initialization
function mlp_new(nn_sizes::Array{Int32,2},ActivationF=Sigmoid)
    nn=MLP(length(nn_sizes),
        map((a,b)->zeros(a,b),nn_sizes[1:end-1],nn_sizes[2:end]),
        map((b)->zeros(b,1),nn_sizes[2:end]),
        ActivationF)
end

# returns the topology of a network as a string
function mlp_get_topology(nnet)
    layer_sizes = [ size(nnet.weights[1],1) ]
    append!(layer_sizes, map(x->size(x,2),nnet.weights))
    layer_sizes = map(x->string(x),layer_sizes)
    join([layer_sizes[1],map(x->join(["x",x]),layer_sizes[2:end])])
end

# display information about the network
function mlp_info(nnet) 
    @printf("MLP - %i layers %s",nnet.num_layers,mlp_get_topology(nnet))
end

## Apply and perturb (modifications to network)

# mlp_apply - applies a function F to a particular network, layer or bias)
#     Inputs: F  - a function Real -> Real 
#             nn - a neural network
#             selected_layers - defaults to all layers,
#                otherwise integer to indicate a particular layer
#                or list of int indices to indicate which layers to apply to 
#    Returns: none
  # mlp_apply_on_bias - variant that only touches the biases
function mlp_apply_on_bias(nnet::MLP,
        F::Function,
        selected_layers=1:nnet.num_layers-1)
    nnet.bias[selected_layers] =
        map(x->map(F,x),nnet.bias[selected_layers])
end

  # mlp_apply_on_bias - variant that only touches the weights
function mlp_apply_on_weights(nnet::MLP, # a network
    F::Function,                          # function to apply
    selected_layers=1:nnet.num_layers-1)  # array to specify which layers 

    nnet.weights[selected_layers] = 
        map(x->map(F,x),nnet.weights[selected_layers])
end

  # mlp_apply - touches both
function mlp_apply(nnet::MLP,
        F::Function,
        selected_layers=1:nnet.num_layers-1)
    mlp_apply_on_bias(nnet,F,selected_layers)
    mlp_apply_on_weights(nnet,F,selected_layers)
end

# mlp_perturb - randomly perturbs the neural network parameters by amount
#    inputs:     nn - a neural network 
#          amountW - maximum amplitude to perturb weights by
#          amountB - maximum amplitude to perturb biases by
#   Returns: None
function mlp_perturb(nnet::MLP,amountW=1,amountB=1)
    fW=x::Float64->x+2*(rand()-0.5)*amountW
    fB=x::Float64->x+2*(rand()-0.5)*amountB
    mlp_apply_on_weights(nnet,fW)
    mlp_apply_on_bias(nnet,fB)
    return nnet
end

# mlp_ff - computes per layer activations for a network given an input
# inputs: NN: a neural network in the structure described
#            x - a set of input vectors
# returns: activation vector
# for _ff1 variant: activations per layer
# for _ff2 variant: activations, followed by input signals per layer
function mlp_ff(nnet::MLP,
    x::Array{Float64,2};
    activationF::Function=Sigmoid)
    @assert size(x,1) == size(nnet.weights[1],1)
    z=x
    l=1
    while l<nnet.num_layers
        z1=nnet.weights[l]'*z
        z1=broadcast(+,nnet.bias[l],z1)
        z = map(activationF, z1)
        l=l+1
    end
    return z
end

# -
function mlp_ff2(nnet::MLP,
    x::Array{Float64,2};
    activationF::Function=Sigmoid)
    A = cell(1,nnet.num_layers)
    Z = cell(1,nnet.num_layers)
    A[1] = x
    l = 1
    while l<nnet.num_layers
        Z[l+1] = nnet.weights[l]'*A[l]
        Z[l+1]=broadcast(+,nnet.bias[l],Z[l+1])
        A[l+1] = map(activationF,Z[l+1])
        l=l+1
    end
    return Z,A
end

# deprecated!
function mlp_ff1(nnet::MLP,

    x::Array{Float64,1};
    activationF::Function=Sigmoid)
    A = cell(1,nnet.num_layers)
    Z = cell(1,nnet.num_layers)
    A[1] = vec(x)
    Z[1] = vec(x)  # layer 1 has no sigmoid
    l = 1
    while l<nnet.num_layers
        Z[l+1] = nnet.weights[l]'*A[l] + nnet.bias[l]
        A[l+1] = map(activationF,Z[l+1])
        l=l+1
    end
    return A,Z
end

# compute the Jacobian
function mlp_compute_J(nnet::MLP,
    t::Array{Float64,2},        # targets
    x::Array{Float64,2},lambda=LAMBDA)        # input vectors

    # compute feed forward stimuli and activations
    (Z,A)=mlp_ff2(nnet,x)

    # calculuate raw J term
    diff = (t-A[nnet.num_layers])
    J=0.5/size(t,2)*sum(vec(diff)'*vec(diff))
        
    # calculate cost from regularization term (sum of magnitude of weights)
    reg = lambda/2.0*sum(map(W->sum(W.*W),nnet.weights))

    # calculate cost function
    J_reg = J+reg
    return J_reg,J,reg
end

# computes the dJ/dW et al for a given input
function mlp_compute_dJ(nnet::MLP,
    t::Array{Float64,2},        # targets
    x::Array{Float64,2})        # input vectors
    
    # compute feed forward stimuli and activations
    (Z,A)=mlp_ff2(nnet,x)
    
    # compute deltas (differential Error) at output layer
    dJ = cell(1,nnet.num_layers)
    l = nnet.num_layers
    dJ[l] = - (t-A[l]) .* (A[l].*(1-A[l]))   # f' for sigmoid

    # back-propagation to compute diff Errors
    while l>1
        l=l-1
        dJ[l] = zeros(size(A[l]))
        dJ[l] = nnet.weights[l]*dJ[l+1] .* (A[l].*(1-A[l]))
    end

    # accumulate the delta errors for all input points
    sum_dJdW=cell(1,nnet.num_layers-1)
    sum_dJdB=cell(1,nnet.num_layers-1)
    for i in 1:nnet.num_layers-1
        sum_dJdB[i] = sum(dJ[i+1],2)
        sum_dJdW[i] = A[i]*dJ[i+1]'
    end

    # normalize by number of inputs
    m=size(t,2)
    for i in 1:nnet.num_layers-1
        sum_dJdW[i] = sum_dJdW[i]/m
        sum_dJdB[i] = sum_dJdB[i]/m
    end

    return sum_dJdW,sum_dJdB,dJ
end

function mlp_gd_update(nnet::MLP,
    t::Array{Float64,2},        # targets
    x::Array{Float64,2};        # input vectors
    alpha=1.0,
    lambda=0.1)
    
    # compute partial differential of Jacobian for the model
    dJdW,dJdB = mlp_compute_dJ(nnet,t,x)
    #print("targest: ",t,"\n")
    for i in 1:nnet.num_layers-1
   #     @printf("Delta %i - ",i)
   #     print(dJdW[i],"\n")
        nnet.weights[i] = (1-alpha*lambda)*nnet.weights[i] - alpha*(dJdW[i])
        nnet.bias[i]    = nnet.bias[i] - alpha*(dJdB[i])
    end
end

# Alternative way to compute the error gradient.
function mlp_compute_dJ2(nnet::MLP,
    t::Array{Float64,2},        # targets
    x::Array{Float64,2})        # input vectors

    dW=map(W->zeros(size(W)),nnet.weights)
    db=map(b->zeros(size(b)),nnet.bias)

    EPSILON=1e-4

    for l in 1:nnet.num_layers-1
        # compute delta-W
        for i in 1:size(nnet.weights[l],1)
            for j in 1:size(nnet.weights[l],2)
                nnet1=deepcopy(nnet)
                nnet2=deepcopy(nnet)
                nnet1.weights[l][i,j] = nnet1.weights[l][i,j] - EPSILON
                nnet2.weights[l][i,j] = nnet2.weights[l][i,j] + EPSILON
                Jr1,J1,reg1 = ANN.mlp_compute_J(nnet1,t,x)
                Jr2,J2,reg2 = ANN.mlp_compute_J(nnet2,t,x)
                dW[l][i,j] = (J2-J1)/(2*EPSILON)
            end
        end
        # compute delta-b
        for i in 1:length(vec(nnet.bias[l]))
            nnet1=deepcopy(nnet)
            nnet2=deepcopy(nnet)
            nnet1.bias[l][i] = nnet1.bias[l][i] - EPSILON
            nnet2.bias[l][i] = nnet2.bias[l][i] + EPSILON
            Jr1,J1,reg1 = ANN.mlp_compute_J(nnet1,t,x)
            Jr2,J2,reg2 = ANN.mlp_compute_J(nnet2,t,x)
            db[l][i] = (J2-J1)/(2*EPSILON)
        end
    end
    return dW,db
end

function get_simple_data_set()
    return [1 1 ; 1 1.2; 0.1 0.1;0 0; 1 -1]',[1 0 0; 1 0 0; 0 1 0;0 1 0; 0 0 1]'
    return [1 1 ; 1 1.2; 0.1 0.1;0 0; 1 -1]',[1 0 0; 1 0 0; 0 1 0;0 1 0; 0 0 1]'
end

# classifies data
function mlp_classify(nnet,data)
    posteriors = mlp_ff(nnet,data) 
    tgts=zeros(size(posteriors,2))
    for k in 1:size(posteriors,2)
        tgts[k] = indmax(posteriors[:,k]) 
    end
    return int(tgts)
end

# computes pc for a data given the answer labels
function mlp_compute_accuracy(nnet,data,labels)
    answers=mlp_classify(nnet,data)
    c=0; w=0
    for i in 1:size(data,2)
        if labels[i] == answers[i]
            c=c+1
        else
            w=w+1
        end
    end
    pc=(100.0*c)/float(c+w)
    return pc,c,w,answers
end

# prints some useful statistics for a current training iteration.
function mlp_report_training_stats(i,net,trnset,trntgts,trnlabels,devset,devlabels)
    ObjF,J,reg = mlp_compute_J(net,trntgts,trnset)
    pc,c,w,answers=mlp_compute_accuracy(net,trnset,trnlabels)
    pc,c,w,answers=mlp_compute_accuracy(net,devset,devlabels)
    @printf("#%4i: %2.3f (%i %i) obj - %3.3f err - %3.3f log reg - %3.3f\n",i,pc,c,w,ObjF,J,log10(reg))
end

###
# MLP training related
###

# Update algorithms 

# Straight-forward back-prop gradient descent
function mlp_bp_update(nnet::MLP,
        trntgts::Array{Float64,2},
        trnset::Array{Float64,2};
        alpha=ALPHA,
        lambda=LAMBDA)
        mlp_gd_update(nnet,trntgts,trnset;alpha=alpha,lambda=lambda)
        return nnet
end

# stochastic gradient descent
function mlp_sgd_update(nnet::MLP,
        trntgts::Array{Float64,2},
        trnset::Array{Float64,2};
        alpha=ALPHA,
        lambda=LAMBDA,
        minibatch=50)

    trnset_size = size(trnset,2)
    sequence=randperm(trnset_size)   # shuffle the training data
    for j in 1:minibatch:trnset_size # run updates in small minibatches
        lastj=min(j+trnset_size,trnset_size)
        selection=sequence[j:lastj]
        mlp_gd_update(nnet,trntgts[:,selection],trnset[:,selection];alpha=alpha,lambda=lambda)
    end
    return nnet
end

##
# Full training algorithm, for convenience
##

# performs supervised training with a training and dev set
function mlp_train_supervised(nnet::MLP,
        trnset::Array{Float64,2},       # training set
        trnlabels::Array{Int32,2},      # training labels
        devset::Array{Float64,2},
        devlabels::Array{Int32,2};
        alg=mlp_bp_update,              # update algorithm - can be
                                        #    mlp_bp_update -- backprop gd
                                        #    mlp_sgd_update -- stochastic gd
        epochs=100,                     # number of training epochs
        num_classes=maximum(trnlabels), # max number of class labels
        alpha=ALPHA,                    # learning rate
        lambda=LAMBDA,
        notifyFunc=mlp_report_training_stats,   
                                        # callback for reporting
        notifyInterval=100)             # reporting interval

    # compute the training targets for this network
    trntgts=expand_labels(trnlabels,num_classes)
    for i in 1:epochs
        alg(nnet,trntgts,trnset;alpha=alpha,lambda=lambda)
        if notifyFunc != None && (i==1 || i%notifyInterval==0)
            notifyFunc(i,nnet,trnset,trntgts,trnlabels,devset,devlabels)
        end
    end
    return nnet
end

####
# Helper functions for Data viewing and generation
####

## return a random sample from a normal (Gaussian) distribution
function rand_normal(mean, stdev)
    if stdev <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand()
    u2 = rand()
    r = sqrt( -2.0*log(u1) )
    theta = 2.0*pi*u2
    mean + stdev*r*sin(theta)
end

## return a random sample from a normal (Gaussian) distribution
function rand_normal2d(stdev)
    if stdev <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand()
    u2 = rand()
    r = sqrt( -2.0*log(u1) )
    theta = 2.0*pi*u2
    [stdev*r*sin(theta),stdev*r*cos(theta)]
end

# generates 2-class labeled data
# returns
function generate_data(data_size,rand_amount)
    ## generate data
    num_labels=3
    num_labels=2
    A=[1 0;0 2;3 3] # collection of mean vectors
    A=[0.8 0.2;0.2 0.8] # collection of mean vectors

    # generate some random labels
    #labels=map(x->(rand()>0.3)?(rand()>0.5)?3:2:1 ,[1:data_size]')
    labels=map(x->(rand()>0.5)?2:1 ,[1:data_size]')
    
    # generate and perturb the data
    targets = expand_labels(labels,num_labels)
    data = A'*targets
    for i in 1:size(data,2)
        data[:,i] = data[:,i]+vec(rand_normal2d(1)*rand_amount)
    end

    return data,labels
end

###
#  Unit Tests
#  NB: I haven't chosen a unit testing framework yet, so for now just call 
#  run_tests
###

function test_sigmoid()
    @test Sigmoid(0) == 0.5
end

function setupNetwork()
    #net=ANN.mlp_perturb(ANN.mlp_new([2 4 3]),1.0,0)
    net=ANN.mlp_perturb(ANN.mlp_new([2 2 2]),1.0,0)
    data=float([ 1 1; 1 0 ; 0 0; 0 1]' )
    labels=[ 1 2 1 2 ]

    return net,data,labels
end

function test_ff() # check that the _ff functions are identical
    net,data,label=setupNetwork()
    Z2,A2=mlp_ff2(net,data)
    #A1=mlp_ff1(net,data)
    y=mlp_ff(net,data)
    @test A2[size(A2,2)] == y  # check that _ff and _ff2 are identical
end

function test_compare_dJ() 
    # checks that backpropagation computation of dJ is similar to 
    # numerical differentiation
    net,data,labels=setupNetwork()
    tgts=expand_labels(labels)

    dW1,dB1,dJ1=mlp_compute_dJ(net,tgts,data)
    dW2,dB2=mlp_compute_dJ2(net,tgts,data)

    @test length(dW1) == length(dW2)
    @test length(dB1) == length(dB2)

    for i in size(dW1,2)
        @test_approx_eq_eps( sum(dW1[i] - dW2[i]), 0 , EPSILON*EPSILON )
        @test_approx_eq_eps( sum(dB1[i] - dB2[i]), 0 , EPSILON*EPSILON )
    end
end

# runs all available tests
function run_tests()
    test_sigmoid()
    test_ff()
    test_compare_dJ()
end

####
# default settings and parameters - nothing for now
####

end # module

# simple test program for the mlp routines
# This example generates two class labels with some fake dataset
function test_mlp(data_size=1000,rand_amount=0.3;alg=ANN.mlp_bp_update)

    # setup for mlp testing
    epochs=10000

    # generate some randomized easily classified data
    data,labels=ANN.generate_data(data_size,rand_amount)
    
    i=int(data_size*0.8)
    j=int(data_size*0.9)
    
    trainset = data[:,1:i];   train_labels = labels[:,1:i]
    devset   = data[:,i:j]; dev_labels = labels[:,i:j]
    testset =  data[:,j:end]; test_labels = labels[:,j:end]
    @printf(" datasets: %i training %i dev %i test examples\n",
            length(train_labels),length(dev_labels),length(test_labels))
    
    # Run Training
    nn=ANN.mlp_perturb(ANN.mlp_new([2 4 3]))
    nn=ANN.mlp_train_supervised(nn,trainset,train_labels,devset,dev_labels;
                num_classes=3,
                alg=alg,
                epochs=10000,
                notifyInterval=100)

    # Compute accuracy and report result
    pc,c,w = ANN.mlp_compute_accuracy(nn,testset,test_labels)
    @printf("  Open PC = %.3f (%i correct, %i wrong)", pc,c,w)
    pc,c,w = ANN.mlp_compute_accuracy(nn,devset,dev_labels)
    @printf("Closed PC = %.3f (%i correct, %i wrong)", pc,c,w)
    pc,c,w = ANN.mlp_compute_accuracy(nn,trainset,train_labels)
    @printf("Closed PC = %.3f (%i correct, %i wrong)", pc,c,w)

    answers=ANN.mlp_classify(nn,data)
    return data,labels,answers,nn
end

