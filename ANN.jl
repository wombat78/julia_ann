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
#   MLP training and computation                                        #
#                                                                       #
# References: http://ufldl.stanford.edu/wiki/index.php/Neural_Networks  #
#                                                                       #
#########################################################################

import Winston;  # required for the plotting functions

module ANN 

ALPHA=1.0 # learning rate
LAMBDA=0.01 # regularization


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

# Tanh

#
# derive - numerically calculates derivatives of functions 
# uses shortcuts if they are known.
function derive(func,delta=1e-10)
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

####
# Improved MLPs, declare a MLP explicit data type
#
#

# Data structure to hold an mlp type -- new functions are created 
# these are uniform networks with only one type neuron with same activation
# accessed as mlp2_FUNC
type MLP
    num_layers  :: Int
    weights     :: Array{ Array {Float64,2 } }
    bias        :: Array{ Array {Float64,2 } }
    activationF :: Function
end

## creation, initialization
function mlp2_new(nn_sizes::Array{Int32,2},ActivationF=Sigmoid)
    nn=MLP(length(nn_sizes),
        map((a,b)->zeros(a,b),nn_sizes[1:end-1],nn_sizes[2:end]),
        map((b)->zeros(b,1),nn_sizes[2:end]),
        ActivationF)
end

# returns the topology of a network as a string
function mlp2_get_topology(nnet)
    layer_sizes = [ size(nnet.weights[1],1) ]
    append!(layer_sizes, map(x->size(x,2),nnet.weights))
    layer_sizes = map(x->string(x),layer_sizes)
    join([layer_sizes[1],map(x->join(["x",x]),layer_sizes[2:end])])
end

# display information about the network
function mlp2_info(nnet) 
    @printf("MLP - %i layers %s",nnet.num_layers,mlp2_get_topology(nnet))
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
function mlp2_apply_on_bias(nnet::MLP,
        F::Function,
        selected_layers=1:nnet.num_layers-1)
    nnet.bias[selected_layers] =
        map(x->map(F,x),nnet.bias[selected_layers])
end

  # mlp_apply_on_bias - variant that only touches the weights
function mlp2_apply_on_weights(nnet::MLP, # a network
    F::Function,                          # function to apply
    selected_layers=1:nnet.num_layers-1)  # array to specify which layers 

    nnet.weights[selected_layers] = 
        map(x->map(F,x),nnet.weights[selected_layers])
end

  # mlp_apply - touches both
function mlp2_apply(nnet::MLP,
        F::Function,
        selected_layers=1:nnet.num_layers-1)
    mlp2_apply_on_bias(nnet,F,selected_layers)
    mlp2_apply_on_weights(nnet,F,selected_layers)
end

# mlp_perturb - randomly perturbs the neural network parameters by amount
#    inputs:     nn - a neural network 
#          amountW - maximum amplitude to perturb weights by
#          amountB - maximum amplitude to perturb biases by
#   Returns: None
function mlp2_perturb(nnet::MLP,amountW=1,amountB=1)
    fW=x::Float64->x+2*(rand()-0.5)*amountW
    fB=x::Float64->x+2*(rand()-0.5)*amountB
    mlp2_apply_on_weights(nnet,fW)
    mlp2_apply_on_bias(nnet,fB)
    return nnet
end

# mlp2_ff - computes per layer activations for a network given an input
# inputs: NN: a neural network in the structure described
#            x - a set of input vectors
# returns: activation vector
# for _ff1 variant: activations per layer
# for _ff2 variant: activations, followed by input signals per layer
function mlp2_ff(nnet::MLP,
    x::Array{Float64,2};
    activationF::Function=Sigmoid)
    @assert size(x,1) == size(nnet.weights[1],1)
    z=x
    l=1
    while l<nnet.num_layers
        z1=nnet.weights[l]'*z
        broadcast(+,nnet.bias[l],z1)
        z = map(activationF, z1)
        l=l+1
    end
    return z
end

# -
function mlp2_ff2(nnet::MLP,
    x::Array{Float64,2};
    activationF::Function=Sigmoid)
    A = cell(1,nnet.num_layers)
    Z = cell(1,nnet.num_layers)
    A[1] = x
    l = 1
    while l<nnet.num_layers
        Z[l+1] = nnet.weights[l]'*A[l]
        broadcast(+,nnet.bias[l],Z[l+1])
        A[l+1] = map(activationF,Z[l+1])
        l=l+1
    end
    return Z,A
end

function mlp2_ff1(nnet::MLP,
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
function mlp2_compute_J(nnet::MLP,
    t::Array{Float64,2},        # targets
    x::Array{Float64,2},lambda=LAMBDA)        # input vectors

    # compute feed forward stimuli and activations
    (Z,A)=mlp2_ff2(nnet,x)

    # calculuate raw J term
    diff = (t-A[nnet.num_layers])
    J=0.5/size(t,2)*sum(vec(diff)'*vec(diff))
        
    # calculate cost from regularization term (sum of magnitude of weights)
    reg = lambda/2.0*sum(sum(map(W->vec(W)'*vec(W),nnet.weights)))

    # calculate cost function
    J_reg = J+reg
    return J_reg,J,reg
end

# computes the dJ/dW et al for a given input
function mlp2_compute_dJ(nnet::MLP,
    t::Array{Float64,2},        # targets
    x::Array{Float64,2})        # input vectors
    
    # compute feed forward stimuli and activations
    (Z,A)=mlp2_ff2(nnet,x)
    
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

function mlp2_gd_update(nnet::MLP,
    t::Array{Float64,2},        # targets
    x::Array{Float64,2},        # input vectors
    alpha=0.1,
    lambda=0.1)
    
    # compute partial differential of Jacobian for the model
    dJdW,dJdB = mlp2_compute_dJ(nnet,t,x)
    for i in 1:nnet.num_layers-1
        nnet.weights[i] = (1-alpha*lambda)*nnet.weights[i] - alpha*(dJdW[i])
        nnet.bias[i]    = (1-alpha*lambda)*nnet.bias[i] - alpha*(dJdB[i])
    end
end

#
#
#
function mlp2_compute_dJ2(nnet::MLP,
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
                Jr1,J1,reg1 = ANN.mlp2_compute_J(nnet1,t,x)
                Jr2,J2,reg2 = ANN.mlp2_compute_J(nnet2,t,x)
                dW[l][i,j] = (J2-J1)/(2*EPSILON)
            end
        end
        # compute delta-b
        for i in 1:length(vec(nnet.bias[l]))
            nnet1=deepcopy(nnet)
            nnet2=deepcopy(nnet)
            nnet1.bias[l][i] = nnet1.bias[l][i] - EPSILON
            nnet2.bias[l][i] = nnet2.bias[l][i] + EPSILON
            J1 = ANN.mlp2_compute_J(nnet1,t,x)[1]
            J2 = ANN.mlp2_compute_J(nnet2,t,x)[1]
            db[l][i] = (J2-J1)/(2*EPSILON)
        end
    end
    print(nnet)
    return dW,db
end

function get_simple_data_set()
    return [1 1 ; 1 1.2; 0.1 0.1;0 0; 1 -1]',[1 0 0; 1 0 0; 0 1 0;0 1 0; 0 0 1]'
    return [1 1 ; 1 1.2; 0.1 0.1;0 0; 1 -1]',[1 0 0; 1 0 0; 0 1 0;0 1 0; 0 0 1]'
end

# classifies data
function mlp2_classify(nnet,data)
    posteriors = mlp2_ff(nnet,data) 
    tgts=zeros(size(posteriors,2))
    for k in 1:size(posteriors,2)
        tgts[k] = indmax(posteriors[:,k]) 
    end
    return int(tgts)
end

function mlp2_compute_accuracy(nnet,data,labels)
    answers=mlp2_classify(nnet,data)
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

function mlp2_bp_train(nnet::MLP,
        trnset::Array{Float64,2},
        trnlabels::Array{Int32,2};
        epochs=100,num_classes=maximum(trnlabels))
err=[]
    pc=[]
    trntgts=mlp_expand_labels(trnlabels,num_classes)
    for i in 1:epochs
        mlp2_gd_update(nnet,trntgts,trnset)
        ObjF,J,reg = mlp2_compute_J(nnet,trntgts,trnset)
        pc,c,w,answers=mlp2_compute_accuracy(nnet,trnset,trnlabels)
        @printf("#%4i: %2.3f (%i %i) obj - %3.3f err - %3.3f reg - %3.3f\n",i,pc,c,w,ObjF,J,reg)
        print ("  " ,answers," vs " , trnlabels,"\n") 
    end
    return nnet
end

####
#
#  Basic Multi-Layer Perceptrons
#
# This is a set of support functions for MLPs, and includes:
#
#    creation/manipulation - mlp_create mlp_apply*
#    feedforward computation - mlp_feedforward
#
# Notes on data structure:
#   we implement a network as a list of (matrix,vector) pairs.
#   matrix(r,c) is the link weight between r'th neuron in previous layer
#   and c'th neuron in next layer, and vector(c) is the bias
#
####

# mlp_create
#   Creates a flat start model for a neural network with given topology.
#    Inputs: a vector indicating number of neurons in each successive layer
#   Returns: an structure containing a neural network, 
#            all weights and biases are initialized to zero.
function mlp_create(nn_sizes)
    return map((a,b)->(zeros(a,b),zeros(b,1)), 
        nn_sizes[1:end-1],nn_sizes[2:end])
end

# mlp_apply - applies a function F to a particular network, layer or bias)
#     Inputs: F  - a function Real -> Real 
#             nn - a neural network
#             selected_layers - defaults to all layers,
#                otherwise integer to indicate a particular layer
#                or list of int indices to indicate which layers to apply to 
#    Returns: none
  # mlp_apply_on_bias - variant that only touches the biases biases
function mlp_apply_on_bias(F,nn,selected_layers=1:length(nn))
    for l in selected_layers
        nn[l] = (nn[l][1],map(F,nn[l][2]))
    end
end

  # mlp_apply_on_bias - variant that only touches the weights
function mlp_apply_on_weights(F,nn,selected_layers=1:length(nn))
    for l in selected_layers
        nn[l] = (map(F,nn[l][1]),nn[l][2])
    end
end

  # mlp_apply - touches both
function mlp_apply(F,nn,selected_layers=1:length(nn))
    mlp_apply_on_bias(F,nn,selected_layers)
    mlp_apply_on_weights(F,nn,selected_layers)
end

# mlp_perturb - randomly perturbs the neural network parameters by amount

# mlp_perturb - randomly perturbs the neural network parameters by amount
#    inputs:     nn - a neural network 
#          amountW - maximum amplitude to perturb weights by
#          amountB - maximum amplitude to perturb biases by
#   Returns: None
function mlp_perturb(nn,amountW=1,amountB=1)
    fW=x->x+2*(rand()-0.5)*amountW
    fB=x->x+2*(rand()-0.5)*amountB
    mlp_apply_on_weights(fW,nn)
    mlp_apply_on_bias(fB,nn)
end

# mlp_check - checks that network is valid
#    inputs: a supposed NN
#   returns: true if ok.
function mlp_check(NN)
    prev_layer_out=0
    try 
    for i in 1:length(NN)
        r,c=size(NN[i][1])
        cv,dim=size(NN[i][2])
        if c != cv  # mismatched bias and weight matrix sizes
            return false
        end
        if dim != 1 # not a correctly oriented vector
            return false
        end
        if prev_layer_out!=0 && prev_layer_out!=r # layer sizes don't match
            return false
        end
        prev_layer_out=cv
    end
    catch
        return false # some generic error
    end
    return true
end

##### Useful functions for feedforward and backprop calculation

# mlp_ff - computes per layer activations for a network given an input
# inputs: NN: a neural network in the structure described
#            x - an input vector
# returns: a vector for final activations
function mlp_ff(NN,x,activationF=Sigmoid)
    y=map(activationF,x)
    A = cell(1,length(NN))
    for i in 1:length(NN)
        y=map(activationF,NN[i][1]'*y + NN[i][2])
        A[i] = y
    end
    return A
end

# mlp_ff2 - J computes the layer input stimulations AND activations 
#           for a network given an input
#   inputs: NN - a neural network in the structure described
#            x - an input vector
#  returns: Z - input stimulus, A - neuron activation
function mlp_ff2(NN,x,activationF=Sigmoid)
    y=map(activationF,x)
    A = cell(1,length(NN))
    Z = cell(1,length(NN))
    for i in 1:length(NN)
        Z[i] = NN[i][1]'*y + NN[i][2]
        A[i] = map(activationF,Z[i])
        y = A[i] # feed output to next layer
    end
    return Z,A
end

# mlp_feedforward
#    computes the final activations for neurons
#   inputs: NN - a neural network in the structure described
#            x - an input vector
#  returns: a vector for final activations
function mlp_feedforward(NN,x,activationF=Sigmoid)
    y=x
    for i in 1:length(NN)
        y=map(activationF,NN[i][1]'*y + NN[i][2])
    end
    return y
end

###
# MLP training related
#

###
# mlp_bp_update1
#
# compute update matrix and vectors for Weights and Biases 
# given a single labeled target. This updates a currently cumulative
# set of update matrices in place
###
function mlp_bp_update1(nn,dW,db,y,x,activationF=Sigmoid)
    # compute derivative of the activation function
    dF = derive(activationF)

    # compute per layer activations
    (Z,A)=mlp_ff2(nn,x,activationF)

    # start from output layer
    deltas = cell(1,length(nn)+1) # errors for whole matrix
    l=length(nn)

    # compute output layer deltas
    y = reshape(y,length(A[l]),1)
    #print(size(y),size(A[l-1]),l," ")
  # deltas[l] = -(y - A[l]).*map(dF,Z[l])
    deltas[l+1] = -(y - A[l]).*(A[l].*(1-A[l]))

    ev = y-A[l] 
    err = 0.5*sum(ev.*ev)
    #print ("\ncompare: ",y,A[l-1])

    # back propagate errors - refer to ufdl.stanford.edu/wiki
    while l>1
       #deltas[l] = nn[l][1]*deltas[l+1].*map(dF,Z[l-1]) 
       #print ("W",size(nn[l+1][1]),"d+1",size(deltas[l+1]),"a",size(1-A[l]),"  \n ")
        deltas[l] = nn[l][1]*deltas[l+1].*(A[l-1].*(1-A[l-1]))
        l=l-1
    end


    # update matrix and vector for all layers
    cx = x
    for l=length(nn):-1:2
#        print(size(dW[l]),"=",size(A[l-1]),"*",size(deltas[l+1]')," ")
        dW[l]=dW[l]+A[l-1]*deltas[l+1]'
        db[l]=db[l]+deltas[l+1]
    end
#    print(size(dW[1]),"=",size(x),"*",size(deltas[2]')," ")
    dW[1]=dW[1]+x*deltas[2]'
    db[1]=db[1]+deltas[2]

    return dW,db,err
end

# mlp_bp_update
#   does batch/mini-batch update given a neural network, 
#   set of labels and inputs
function mlp_bp_update(nn,y,x,activationF=Sigmoid,alpha=ALPHA,lambda=LAMBDA) 

    m = size(y,2)  # get number of number of training examples

    # initialize a pair of variables to hold the update parameters
    dW=cell(1,length(nn))
    db=cell(1,length(nn))
    for l in 1:length(nn)
        dW[l] = zeros(size(nn[l][1]))
        db[l] = zeros(size(nn[l][2]))
    end
    
    # cumulate the weight updates over the minibatch
    cum_err = 0
    reg_err = 0
    for i in 1:size(y,2)
        #print ("lastest", size(y[:,i]),size(x[:,i]),size(nn[1][1]))
        (dW,db,err)=mlp_bp_update1(nn,dW,db,y[:,i],x[:,i],activationF)
        cum_err=cum_err+err
    end
    cum_err=cum_err/size(y,2)

    for i in 1:length(nn)
        reg_err = reg_err + lambda/2*sum(nn[i][1].*nn[i][1])
    end 
    J = cum_err + reg_err

    ## # DEBUG- @printf(" %i samples -- alpha %f ERR %f dW mag = %.4f \n",m,alpha,cum_err ,sum(map(x->sum(x.*x),dW)))
    @printf(" %i samples -- alpha %f ERR %f reg =%f J= %f dW mag = %.4f \n",m,alpha,cum_err ,reg_err,J,sum(map(x->sum(x.*x),dW)))

    # update the neural network weight vectors for this minibatch
    for i in 1:length(nn)
                           #  update matrix , regularization term
        W = nn[i][1]
        b = nn[i][2]

        W1 = W - alpha * ((dW[i]/m) + lambda*W ) 
        #print (size(b),size(db[i]))
        b1 = b - alpha * (db[i]/m) 

        nn[i] = (W1,b1)

        #nn[i] = (nn[i][1] -                 
        #             alpha*((dW[i]/m) + lambda*nn[i][1])  ,
        #            nn[i][2] - alpha*(db[i]/m))  # bias vectors
        #nn[i] = (nn[i][1] ,
        #        nn[i][2] - alpha*(db[i]/m) ) # bias vectors
    end
    return dW,db,cum_err

end


function magnitude(x)
    return sqrt(sum(x.*x))
end

function mlp_jacobian(nn,input,tgt)
    return 0.5*magnitude(mlp_feedforward(nn,input)-tgt)
end

# takes a list of labels (int), and expands it to vector format
#   * useful for training classifiers
function mlp_expand_labels(labels,last_label=0)
    if last_label == 0
        last_label=max(last_label,maximum(labels))
    end
    targets=zeros(last_label,length(labels))
    for i in 1:length(labels)
        targets[labels[i],i] = 1 
    end
    return targets
end

function mlp_weight_sq_sum(nn)
    return sum(map(x->sum(x[1]*x[1]),nn))
end

# create an appropriate ann based on the data input and labels
function mlp_create_nn(input_dim,num_classes,topology=[4])
    # initialize the neural network
    #  --  create a neural network and do randomized flat start init
    topo = [ input_dim ]
    append!(topo,reshape(topology,length(topology)))
    append!(topo,[num_classes])

    ANN.mlp_create(topo)
end

# Performs standard ANN full-batch training from flat start initialization
#  Inputs: trnset,trn_labels - labelled training set
#          devset,dev_labels - labelled training set
#          epochs - number of iterations to train
#          topology - number of hidden nodes, as a list
#          alpha - learning rate
#          lambda - regularization term
# Returns: trained neural network
function mlp_bp_train(nn,trnset,trn_labels,devset,dev_labels,activationF=Sigmoid,epochs=100,topology=[6],alpha=ALPHA,lambda=0.1)

    input_dim  = size(trnset,1)
    num_classes = max(maximum(trn_labels),maximum(dev_labels))
    

    if nn==None
        @printf("Starting training for %i-dim input, %i target classes\n",
              input_dim,num_classes)
        nn=mlp_create_nn(input_dim,num_classes,topology)
    else
        @printf("Continuing training for %i-dim input, %i target classes\n",
              input_dim,num_classes)
    end
    @printf("  update rate: %.3f, regularization: %.3f\n",alpha,lambda)

    ANN.mlp_perturb(nn,1,0)
    t=ANN.mlp_classify(nn,devset)
    print("counts:" ,map(x->count(j->j==x,t),1:num_classes))
    minC=minimum(map(x->count(j->j==x,t),1:num_classes))

    print("Min Count!",minC)
    while minC==0
        ANN.mlp_perturb(nn,0.1,0.1)
        t=ANN.mlp_classify(nn,devset)
    #print("counts:" ,map(x->count(j->j==x,t),1:num_classes))
        minC=minimum(map(x->count(j->j==x,t),1:num_classes))
    #    print("Min Count!",minC)
    end

    print(nn)

    #  --  Run training
    trn_targets = mlp_expand_labels(trn_labels,num_classes)
    errv=zeros(1,epochs)
    for i in 1:epochs
        #ANN.mlp_perturb(nn,alpha)
        #alpha=alpha*0.9
        #if alpha<0.01 
        #    alpha = 0.01
        #end
#
        dW,dB,errv[i]=mlp_bp_update(nn,trn_targets,trnset,activationF,alpha,lambda)

        update_magnitude = [map(x->sum(x.*x),dW),map(x->sum(x.*x),dB)]

        c,w,pc=mlp_compute_accuracy(nn,devset,dev_labels)

        @printf("#%i - accuracy %.3f (%i correct %i wrong)\n",i,pc,c,w)
        for i in 1:length(update_magnitude)
            @printf(" %.3f" ,update_magnitude[i])
        end
        @printf("\n")
    end
    return nn,errv
end


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
function rand_normal2d( stdev)
    if stdev <= 0.0
        error("standard deviation must be positive")
    end
    u1 = rand()
    u2 = rand()
    r = sqrt( -2.0*log(u1) )
    theta = 2.0*pi*u2
    [stdev*r*sin(theta),stdev*r*cos(theta)]
end

###
# classification and testing routines

# mlp_classify compute classification result
function mlp_classify(nn,data)
    tgts = zeros(1,size(data,2))
    for i in 1:size(data,2)
        y= ANN.mlp_feedforward(nn,data[:,i]) 
        A=ANN.mlp_ff(nn,data[:,i])
        tgts[i]=indmax(y)
        if maximum(y)<0.1
            #print ("Activation: %.3f\n",y)
            tgts[i] = 0
        end
           # @printf("Activation: %.3f to %.3f\n",maximum(y),minimum(y))
           # @printf("          : %.3f to %.3f\n",maximum(A[length[A]]),minimum(A[length[A]]))
    end
    return tgts
end

function mlp_compute_accuracy(nn,data,labels)
    correct=0; wrong=0
    tgts=mlp_classify(nn,data)
    #print( length(labels))
    #print( length(tgts))
    for i=1:length(labels)
        if labels[i] == tgts[i]
            correct=correct+1
        else
            wrong=wrong+1
        end
    end
    percent_correct = (100.0*correct)/(correct+wrong)
    return correct,wrong,percent_correct
end


####
# Helper functions for Data viewing and generation
####

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
    targets = mlp_expand_labels(labels,num_labels)
    data = A'*targets
    for i in 1:size(data,2)
        data[:,i] = data[:,i]+vec(rand_normal2d(1)*rand_amount)
    end

    return data,labels
end

# generates 3-class labeled data
# returns
function plot_data(data,labels,doplot=plot)
    num_labels=maximum(labels)
    colors="roygbv"
    for i in 1:num_labels
        class_idx = find(map(x->x==i,labels))
        doplot(data[1,class_idx,1],data[2,class_idx],string(colors[i]))
    end
end

####
# default settings and parameters
####
mlp_train = mlp_bp_train        # use full-batch training


end # module

# simple test program for the mlp routines
# This example generates two class labels with some fake dataset
function test_mlp(data_size=100,rand_amount=0.1)

    # setup for mlp testing
    epochs=10

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
    nn,err=ANN.mlp_train(None,trainset,train_labels,
                     devset,dev_labels,
                     ANN.Sigmoid,epochs,[5 4])

    # Compute accuracy and report result
    c,w,pc = ANN.mlp_compute_accuracy(nn,testset,test_labels)
    @printf("  Open PC = %.3f (%i correct, %i wrong)", pc,c,w)
    c,w,pc = ANN.mlp_compute_accuracy(nn,devset,dev_labels)
    @printf("Closed PC = %.3f (%i correct, %i wrong)", pc,c,w)
    c,w,pc = ANN.mlp_compute_accuracy(nn,trainset,train_labels)
    @printf("Closed PC = %.3f (%i correct, %i wrong)", pc,c,w)

    answers=ANN.mlp_classify(nn,data)
    return data,labels,answers,nn,err

end


# test case - check consistency between neural net implementations
function test_mlp_ff_fn(nn,data)
    for i in 1:size(x,2)
        Y=ANN.mlp_feedforward(nn,data[:,i])
        A1=ANN.mlp_ff(nn,data[:,i])
        Z2,A2=ANN.mlp_ff2(nn,data[:,i])
        if Y!=A[length(nn)] || A1[length(nn)]!=A2[length(nn)] 
            return false
        end
    end
    return true
end


function PLOT()
reload("ANN.jl")
@time data,label,answers,nn,err=test_mlp();

c1=find(x->x==1,answers);
c2=find(x->x==2,answers);
plot(data[1,c1],data[2,c1],"ro",data[1,c2],data[2,c2],"go")


end
