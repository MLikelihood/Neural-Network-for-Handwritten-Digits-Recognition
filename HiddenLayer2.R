
##Train Data set, and preprocessing input scaling 

train=read.csv("C:\\Users\\shuzhang\\Downloads\\optdigits.tra",header=FALSE)

#create Data Partition : training set + validation set using training data
training <- train[1:3058, ]
CV <- train[3059:3823,]

X <- as.matrix(training[, -65]) 
# data matrix (each row = single example)
N <- nrow(X)
# number of examples
y <- training[,65] 
# class labels 
K <- length(unique(y)) 
# number of classes
X.proc <- X/max(X)
# scale
D <- ncol(X.proc)
# dimensionality 
Xcv <- as.matrix(CV[, -65]) 
# data matrix (each row = single example)
ycv <- CV[, 65] 
# class labels
Xcv.proc <- Xcv/max(X) 
# scale CV data 
Y <- matrix(0, N, K) 
for (i in 1:N){ 
  Y[i, y[i]+1]= 1
}

Y_V <- matrix(0, nrow(Xcv.proc),length(unique(ycv)) ) 
for (i in 1:nrow(Xcv.proc)){ 
  Y_V[i, ycv[i]+1]= 1
}


## read test set
test=read.csv("C:\\Users\\shuzhang\\Desktop\\optdigits.tes",header=FALSE)

XT <- as.matrix(test[, -65]) 
# data matrix (each row = single example)
NT <- nrow(X)
# number of examples
yT <- test[,65] 
# class labels 
KT <- length(unique(y)) 
# number of classes
XT.proc <- XT/max(XT)
# scale
DT <- ncol(X.proc)
# dimensionality 

###################################################################################################################

##ReLU hidden units,cross-entropy loss,hidden layers=2,hidden units in each layer=10, 
##learning rates=0.5, momentum rates= 0.05, regulation=0.001

nnet <- function(X, Y,Xcv.proc, Y_V,reg = 0.0001,learningRate,  h, m, niteration){ 
  # get dim of input 
  N <- nrow(X) 
  # number of examples  
  K <- ncol(Y) 
  # number of classes 
  D <- ncol(X) # dimensionality  
  # initialize parameters randomly  
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)  
  b <- matrix(0, nrow = 1, ncol = h)  
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h) 
  b2 <- matrix(0, nrow = 1, ncol = K) 
  W3 <- 0.01 * matrix(rnorm(h*K), nrow = h) 
  b3 <- matrix(0, nrow = 1, ncol = K)
  validation_error=vector('numeric')
  
  # gradient descent loop to update weight and bias  
  for (i in 0:niteration){     
    # hidden layer 1, ReLU activation   
    hidden_layer1 <- tanh(X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  
    hidden_layer1 <- matrix(hidden_layer1, nrow = N)   
    # hidden layer 2, ReLU activation   
    hidden_layer2 <- tanh(hidden_layer1%*% W2 + matrix(rep(b2,N), nrow = N, byrow = T))  
    hidden_layer2 <- matrix(hidden_layer2, nrow = N)   
    # class score   
    scores <- hidden_layer2%*%W3 + matrix(rep(b3,N), nrow = N, byrow = T)     
    # compute and normalize class probabilities    
    exp_scores <- exp(scores)   
    probs <- exp_scores / rowSums(exp_scores)    
    # compute the cross-entropy loss: sofmax and regularization   
    corect_logprobs <- -log(probs)   
    data_loss <- sum(corect_logprobs*Y)/N   
    reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) +0.5*reg*sum(W3*W3)  
    loss <- data_loss + reg_loss   
    
    # compute the gradient on scores   
    dscores <- probs-Y   
    dscores <- dscores/N    
    
    # backpropate the gradient to the parameters   
    dW3 <- t(hidden_layer2)%*%dscores   
    db3 <- colSums(dscores)   
    
    # modify with momentum m
    dW3 <-dW3+(dW3)*m
    db3 <-db3+(db3)*m
    
    # next backprop into hidden layer 2
    dhidden2 <- dscores%*%t(W3)  
    # backprop the ReLU non-linearity  
  #  dhidden2[hidden_layer2 <= 0] <- 0   
    
    
    # backpropate the gradient to the parameters   
    dW2 <- t(hidden_layer1)%*%dhidden2   
    db2 <- colSums(dhidden2)   
    
    # modify with momentum m
    dW2 <-dW2+(dW2)*m
    db2 <-db2+(db2)*m
    
    # next backprop into hidden layer  
    dhidden1 <- dscores%*%t(W2)  
    # backprop the ReLU non-linearity  
    dhidden1[hidden_layer1 <= 0] <- 0   
    # finally into W,b   
    dW <- t(X)%*%dhidden1   
    db <- colSums(dhidden1) 
    
    # modify with momentum m
    dW <-dW+(dW)*m
    db <-db+(db)*m
    
    # add regularization gradient contribution 
    dW3 <- dW3 + reg *W3
    dW2 <- dW2 + reg *W2    
    dW <- dW + reg *W   
    
    # update parameter   
    W <- W-learningRate*dW    
    b <- b-learningRate*db   
    W2 <- W2-learningRate*dW2   
    b2 <- b2-learningRate*db2 
    W3 <- W3-learningRate*dW3   
    b3 <- b3-learningRate*db3   
    
    # check progress 
    if (i%%50 == 0 | i == niteration){ 
      values=0
      N_V=nrow(Xcv.proc)
      for(j in 1:length(ycv)){
        # values=values+(max((((matrix(Xcv.proc[j,],1,64)%*%W+b)%*%W2)+b2))-ycv[j]-1)^2
        hidden_layer1_V <- tanh(Xcv.proc%*% W + matrix(rep(b,N_V), nrow = N_V, byrow = T))  
        hidden_layer1_V <- matrix(hidden_layer1_V, nrow = N_V) 
        hidden_layer2_V <- tanh(hidden_layer1_V%*% W2 + matrix(rep(b2,N_V), nrow = N_V, byrow = T))  
        hidden_layer2_V <- matrix(hidden_layer2_V, nrow = N_V) 
        scores_V <- hidden_layer2_V%*%W3+ matrix(rep(b3,N_V), nrow = N_V, byrow = T)     
        # compute and normalize class probabilities    
        exp_scores_V <- exp(scores_V)   
        probs_V <- exp_scores_V / rowSums(exp_scores_V)    
        # compute the cross-entropy loss: sofmax and regularization   
        corect_logprobs_V <- -log(probs_V)   
        data_loss_V <- sum(corect_logprobs_V*Y_V)/N_V   
        reg_loss_V <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) +0.5*reg*sum(W3*W3)
        loss_V <- data_loss_V + reg_loss_V  
        
      }
      validation_error <- c(validation_error, loss_V)
       #print(paste("iteration", i,': train loss', loss))#
       #print(paste("iteration", i,': validation error', validation_error[i%/%50+1]))#
      if(is.unsorted(rev(validation_error[-c(1,2)]))==TRUE){
        break
      }  
       
    }
    
  }
  
  return(list(W, b, W2, b2,W3,b3))}

##predict function 
nnetPred <- function(X, para = list()){  
  W <- para[[1]]  
  b <- para[[2]] 
  W2 <- para[[3]] 
  b2 <- para[[4]]
  W3 <- para[[5]] 
  b3 <- para[[6]] 
  N <- nrow(X) 
 # hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  
  #hidden_layer <- matrix(hidden_layer, nrow = N) 
 # scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)  
  
  hidden_layer1 <- tanh(X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  
  hidden_layer1 <- matrix(hidden_layer1, nrow = N)   
  # hidden layer 2, ReLU activation   
  hidden_layer2 <- tanh( hidden_layer1%*% W2 + matrix(rep(b2,N), nrow = N, byrow = T))  
  hidden_layer2 <- matrix(hidden_layer2, nrow = N)   
  # class score   
  scores <- hidden_layer2%*%W3 + matrix(rep(b3,N), nrow = N, byrow = T) 
  predicted_class <- apply(scores, 1, which.max)   
  return(predicted_class)  }

##class accuracy
class_accuracy=function(x){
  class_accuracy=vector("numeric")
  for(i in 1:10){
    accuracy=diag(x)[i]/sum(x[,i])
    class_accuracy=c(class_accuracy,accuracy)
  }
  class_accuracy
}

##Performance of training set
ptm <- proc.time()
nnet.mnist <- nnet(X.proc, Y, Xcv.proc, Y_V,learningRate = 0.3, h = 10, m=0.05, niteration= 6000)
proc.time() - ptm
predicted_class <- nnetPred(X.proc, nnet.mnist)
print(paste('training set accuracy:', mean(predicted_class == (y+1))))
h=cbind(as.vector(y+1),as.vector(predicted_class)) 
confusionMatrix = table (h[,1], h[,2])
print(confusionMatrix)
class_accuracy(confusionMatrix)

##Performance of test  set
predicted_class_Test <- nnetPred(XT.proc, nnet.mnist)
print(paste('test set accuracy:',   mean(predicted_class_Test == (yT+1))))
h_t=cbind(as.vector(yT+1),as.vector(predicted_class_Test)) 
confusionMatrix_Test = table (h_t[,1], h_t[,2])
print(confusionMatrix_Test)
class_accuracy(confusionMatrix_Test)


###################################################################
#########tanh+Cross Entropy
nnet <- function(X, Y,Xcv.proc, Y_V,reg = 0.0001,learningRate,  h, m, niteration){ 
  # get dim of input 
  N <- nrow(X) 
  # number of examples  
  K <- ncol(Y) 
  # number of classes 
  D <- ncol(X) # dimensionality  
  # initialize parameters randomly  
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)  
  b <- matrix(0, nrow = 1, ncol = h)  
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h) 
  b2 <- matrix(0, nrow = 1, ncol = K) 
  W3 <- 0.01 * matrix(rnorm(h*K), nrow = h) 
  b3 <- matrix(0, nrow = 1, ncol = K)
  validation_error=vector('numeric')
  
  # gradient descent loop to update weight and bias  
  for (i in 0:niteration){     
    # hidden layer 1, ReLU activation   
    hidden_layer1 <- tanh(X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  
    hidden_layer1 <- matrix(hidden_layer1, nrow = N)   
    # hidden layer 2, ReLU activation   
    hidden_layer2 <- tanh(hidden_layer1%*% W2 + matrix(rep(b2,N), nrow = N, byrow = T))  
    hidden_layer2 <- matrix(hidden_layer2, nrow = N)   
    # class score   
    scores <- hidden_layer2%*%W3 + matrix(rep(b3,N), nrow = N, byrow = T)     
    # compute and normalize class probabilities    
    exp_scores <- exp(scores)   
    probs <- exp_scores / rowSums(exp_scores)    
    # compute the cross-entropy loss: sofmax and regularization   
    corect_logprobs <- -log(probs)   
    data_loss <- sum(corect_logprobs*Y)/N   
    reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) +0.5*reg*sum(W3*W3)  
    loss <- data_loss + reg_loss   
    
    # compute the gradient on scores   
    dscores <- probs-Y   
    dscores <- dscores/N    
    
    # backpropate the gradient to the parameters   
    dW3 <- t(hidden_layer2)%*%dscores   
    db3 <- colSums(dscores)   
    
    # modify with momentum m
    dW3 <-dW3+(dW3)*m
    db3 <-db3+(db3)*m
    
    # next backprop into hidden layer 2
    dhidden2 <- dscores%*%t(W3)  
    # backprop the ReLU non-linearity  
    #  dhidden2[hidden_layer2 <= 0] <- 0   
    
    
    # backpropate the gradient to the parameters   
    dW2 <- t(hidden_layer1)%*%dhidden2   
    db2 <- colSums(dhidden2)   
    
    # modify with momentum m
    dW2 <-dW2+(dW2)*m
    db2 <-db2+(db2)*m
    
    # next backprop into hidden layer  
    dhidden1 <- dscores%*%t(W2)  
    # backprop the ReLU non-linearity  
    dhidden1[hidden_layer1 <= 0] <- 0   
    # finally into W,b   
    dW <- t(X)%*%dhidden1   
    db <- colSums(dhidden1) 
    
    # modify with momentum m
    dW <-dW+(dW)*m
    db <-db+(db)*m
    
    # add regularization gradient contribution 
    dW3 <- dW3 + reg *W3
    dW2 <- dW2 + reg *W2    
    dW <- dW + reg *W   
    
    # update parameter   
    W <- W-learningRate*dW    
    b <- b-learningRate*db   
    W2 <- W2-learningRate*dW2   
    b2 <- b2-learningRate*db2 
    W3 <- W3-learningRate*dW3   
    b3 <- b3-learningRate*db3   
    
    # check progress 
    if (i%%50 == 0 | i == niteration){ 
      values=0
      N_V=nrow(Xcv.proc)
      for(j in 1:length(ycv)){
        # values=values+(max((((matrix(Xcv.proc[j,],1,64)%*%W+b)%*%W2)+b2))-ycv[j]-1)^2
        hidden_layer1_V <- tanh(Xcv.proc%*% W + matrix(rep(b,N_V), nrow = N_V, byrow = T))  
        hidden_layer1_V <- matrix(hidden_layer1_V, nrow = N_V) 
        hidden_layer2_V <- tanh(hidden_layer1_V%*% W2 + matrix(rep(b2,N_V), nrow = N_V, byrow = T))  
        hidden_layer2_V <- matrix(hidden_layer2_V, nrow = N_V) 
        scores_V <- hidden_layer2_V%*%W3+ matrix(rep(b3,N_V), nrow = N_V, byrow = T)     
        # compute and normalize class probabilities    
        exp_scores_V <- exp(scores_V)   
        probs_V <- exp_scores_V / rowSums(exp_scores_V)    
        # compute the cross-entropy loss: sofmax and regularization   
        corect_logprobs_V <- -log(probs_V)   
        data_loss_V <- sum(corect_logprobs_V*Y_V)/N_V   
        reg_loss_V <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) +0.5*reg*sum(W3*W3)
        loss_V <- data_loss_V + reg_loss_V  
        
      }
      validation_error <- c(validation_error, loss_V)
      #print(paste("iteration", i,': train loss', loss))#
      #print(paste("iteration", i,': validation error', validation_error[i%/%50+1]))#
      if(is.unsorted(rev(validation_error[-c(1,2,3)]))==TRUE){
        break
      }  
      
    }
    
  }
  
  return(list(W, b, W2, b2,W3,b3))}

##predict function 
nnetPred <- function(X, para = list()){  
  W <- para[[1]]  
  b <- para[[2]] 
  W2 <- para[[3]] 
  b2 <- para[[4]]
  W3 <- para[[5]] 
  b3 <- para[[6]] 
  N <- nrow(X) 
  # hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  
  #hidden_layer <- matrix(hidden_layer, nrow = N) 
  # scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)  
  
  hidden_layer1 <- tanh(X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  
  hidden_layer1 <- matrix(hidden_layer1, nrow = N)   
  # hidden layer 2, ReLU activation   
  hidden_layer2 <- tanh( hidden_layer1%*% W2 + matrix(rep(b2,N), nrow = N, byrow = T))  
  hidden_layer2 <- matrix(hidden_layer2, nrow = N)   
  # class score   
  scores <- hidden_layer2%*%W3 + matrix(rep(b3,N), nrow = N, byrow = T) 
  predicted_class <- apply(scores, 1, which.max)   
  return(predicted_class)  }

##class accuracy
class_accuracy=function(x){
  class_accuracy=vector("numeric")
  for(i in 1:10){
    accuracy=diag(x)[i]/sum(x[,i])
    class_accuracy=c(class_accuracy,accuracy)
  }
  class_accuracy
}

##Performance of training set
ptm <- proc.time()
nnet.mnist <- nnet(X.proc, Y, Xcv.proc, Y_V,learningRate = 0.3, h = 10, m=0.05, niteration= 6000)
proc.time() - ptm
predicted_class <- nnetPred(X.proc, nnet.mnist)
print(paste('training set accuracy:', mean(predicted_class == (y+1))))
h=cbind(as.vector(y+1),as.vector(predicted_class)) 
confusionMatrix = table (h[,1], h[,2])
print(confusionMatrix)
class_accuracy(confusionMatrix)

##Performance of test  set
predicted_class_Test <- nnetPred(XT.proc, nnet.mnist)
print(paste('test set accuracy:',   mean(predicted_class_Test == (yT+1))))
h_t=cbind(as.vector(yT+1),as.vector(predicted_class_Test)) 
confusionMatrix_Test = table (h_t[,1], h_t[,2])
print(confusionMatrix_Test)
class_accuracy(confusionMatrix_Test)















#######################################################################################################

#Sigmoid+ Cross Entropy


nnet <- function(X, Y,Xcv.proc, Y_V,reg = 0.0001,learningRate,  h, m, niteration){ 
  # get dim of input 
  N <- nrow(X) 
  # number of examples  
  K <- ncol(Y) 
  # number of classes 
  D <- ncol(X) # dimensionality  
  # initialize parameters randomly  
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)  
  b <- matrix(0, nrow = 1, ncol = h)  
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h) 
  b2 <- matrix(0, nrow = 1, ncol = K) 
  W3 <- 0.01 * matrix(rnorm(h*K), nrow = h) 
  b3 <- matrix(0, nrow = 1, ncol = K)
  validation_error=vector('numeric')
  
  # gradient descent loop to update weight and bias  
  for (i in 0:niteration){     
    hidden_layer1 <- sigmoid(X%*% W + matrix(rep(b,N), nrow = N, byrow = T),SoftMax=TRUE)  
    hidden_layer1 <- matrix(hidden_layer1, nrow = N)   

    hidden_layer2 <- sigmoid(hidden_layer1%*% W2 + matrix(rep(b2,N), nrow = N, byrow = T),SoftMax=TRUE)  
    hidden_layer2 <- matrix(hidden_layer2, nrow = N)   
    # class score   
    scores <- hidden_layer2%*%W3 + matrix(rep(b3,N), nrow = N, byrow = T)     
    # compute and normalize class probabilities    
    exp_scores <- exp(scores)   
    probs <- exp_scores / rowSums(exp_scores)    
    # compute the cross-entropy loss: sofmax and regularization   
    corect_logprobs <- -log(probs)   
    data_loss <- sum(corect_logprobs*Y)/N   
    reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) +0.5*reg*sum(W3*W3)  
    loss <- data_loss + reg_loss   
    
    # compute the gradient on scores   
    dscores <- probs-Y   
    dscores <- dscores/N    
    
    # backpropate the gradient to the parameters   
    dW3 <- t(hidden_layer2)%*%dscores   
    db3 <- colSums(dscores)   
    
    # modify with momentum m
    dW3 <-dW3+(dW3)*m
    db3 <-db3+(db3)*m
    
    # next backprop into hidden layer 2
    dhidden2 <- dscores%*%t(W3)  
    # backprop the ReLU non-linearity  
    dhidden2[hidden_layer2 <= 0] <- 0   
    
    
    # backpropate the gradient to the parameters   
    dW2 <- t(hidden_layer1)%*%dhidden2   
    db2 <- colSums(dhidden2)   
    
    # modify with momentum m
    dW2 <-dW2+(dW2)*m
    db2 <-db2+(db2)*m
    
    # next backprop into hidden layer  
    dhidden1 <- dscores%*%t(W2)  
    # backprop the ReLU non-linearity  
    dhidden1[hidden_layer1 <= 0] <- 0   
    # finally into W,b   
    dW <- t(X)%*%dhidden1   
    db <- colSums(dhidden1) 
    
    # modify with momentum m
    dW <-dW+(dW)*m
    db <-db+(db)*m
    
    # add regularization gradient contribution 
    dW3 <- dW3 + reg *W3
    dW2 <- dW2 + reg *W2    
    dW <- dW + reg *W   
    
    # update parameter   
    W <- W-learningRate*dW    
    b <- b-learningRate*db   
    W2 <- W2-learningRate*dW2   
    b2 <- b2-learningRate*db2 
    W3 <- W3-learningRate*dW3   
    b3 <- b3-learningRate*db3   
    
    # check progress 
    if (i%%50 == 0 | i == niteration){ 
      values=0
      N_V=nrow(Xcv.proc)
      for(j in 1:length(ycv)){
        # values=values+(max((((matrix(Xcv.proc[j,],1,64)%*%W+b)%*%W2)+b2))-ycv[j]-1)^2
        hidden_layer1_V <- sigmoid(Xcv.proc%*% W + matrix(rep(b,N_V), nrow = N_V, byrow = T),SoftMax=TRUE)  
        hidden_layer1_V <- matrix(hidden_layer1_V, nrow = N_V) 
        hidden_layer2_V <- sigmoid(hidden_layer1_V%*% W2 + matrix(rep(b2,N_V), nrow = N_V, byrow = T),SoftMax=TRUE)  
        hidden_layer2_V <- matrix(hidden_layer2_V, nrow = N_V) 
        
        scores_V <- hidden_layer2_V%*%W3+ matrix(rep(b3,N_V), nrow = N_V, byrow = T)   
        
        # compute and normalize class probabilities    
        exp_scores_V <- exp(scores_V)   
        
        probs_V <- exp_scores_V / rowSums(exp_scores_V) 
        
        # compute the cross-entropy loss: sofmax and regularization   
        corect_logprobs_V <- -log(probs_V)  
        
        data_loss_V <- sum(corect_logprobs_V*Y_V)/N_V   

        reg_loss_V <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) +0.5*reg*sum(W3*W3)
        loss_V <- data_loss_V + reg_loss_V  
        
        
      }
      validation_error <- c(validation_error, loss_V)
      #print(paste("iteration", i,': train loss', loss))#
      #print(paste("iteration", i,': validation error', validation_error[i%/%50+1]))#
      if(is.unsorted(rev(validation_error))==TRUE){
        break
      }  
      
    }
    
  }
  
  return(list(W, b, W2, b2,W3,b3))}

##predict function 
nnetPred <- function(X, para = list()){  
  W <- para[[1]]  
  b <- para[[2]] 
  W2 <- para[[3]] 
  b2 <- para[[4]]
  W3 <- para[[5]] 
  b3 <- para[[6]] 
  N <- nrow(X) 
  # hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  
  #hidden_layer <- matrix(hidden_layer, nrow = N) 
  # scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)  
  
  hidden_layer1 <- sigmoid(X%*% W + matrix(rep(b,N), nrow = N, byrow = T),SoftMax=TRUE)  
  hidden_layer1 <- matrix(hidden_layer1, nrow = N)   
  # hidden layer 2, ReLU activation   
  hidden_layer2 <- sigmoid( hidden_layer1%*% W2 + matrix(rep(b2,N), nrow = N, byrow = T),SoftMax=TRUE)  
  hidden_layer2 <- matrix(hidden_layer2, nrow = N)   
  # class score   
  scores <- hidden_layer2%*%W3 + matrix(rep(b3,N), nrow = N, byrow = T) 
  predicted_class <- apply(scores, 1, which.max)   
  return(predicted_class)  }

##class accuracy
class_accuracy=function(x){
  class_accuracy=vector("numeric")
  for(i in 1:10){
    accuracy=diag(x)[i]/sum(x[,i])
    class_accuracy=c(class_accuracy,accuracy)
  }
  class_accuracy
}

##Performance of training set
ptm <- proc.time()
nnet.mnist <- nnet(X.proc, Y, Xcv.proc, Y_V,learningRate = 0.1, h = 10, m=0.05, niteration= 6000)
proc.time() - ptm
predicted_class <- nnetPred(X.proc, nnet.mnist)
print(paste('training set accuracy:', mean(predicted_class == (y+1))))
h=cbind(as.vector(y+1),as.vector(predicted_class)) 
confusionMatrix = table (h[,1], h[,2])
print(confusionMatrix)
class_accuracy(confusionMatrix)

##Performance of test  set
predicted_class_Test <- nnetPred(XT.proc, nnet.mnist)
print(paste('test set accuracy:',   mean(predicted_class_Test == (yT+1))))
h_t=cbind(as.vector(yT+1),as.vector(predicted_class_Test)) 
confusionMatrix_Test = table (h_t[,1], h_t[,2])
print(confusionMatrix_Test)
class_accuracy(confusionMatrix_Test)





###########################################################################################
#Sigmoid+Sum of Squares

nnet <- function(X, Y,Xcv.proc, Y_V,reg = 0.0001,learningRate,  h, m, niteration){ 
  # get dim of input 
  N <- nrow(X) 
  # number of examples  
  K <- ncol(Y) 
  # number of classes 
  D <- ncol(X) # dimensionality  
  # initialize parameters randomly  
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)  
  b <- matrix(0, nrow = 1, ncol = h)  
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h) 
  b2 <- matrix(0, nrow = 1, ncol = K) 
  W3 <- 0.01 * matrix(rnorm(h*K), nrow = h) 
  b3 <- matrix(0, nrow = 1, ncol = K)
  validation_error=vector('numeric')
  
  # gradient descent loop to update weight and bias  
  for (i in 0:niteration){     
    # hidden layer 1, ReLU activation   
    hidden_layer1 <- sigmoid(X%*% W + matrix(rep(b,N), nrow = N, byrow = T),SoftMax=TRUE)  
    hidden_layer1 <- matrix(hidden_layer1, nrow = N)   
    # hidden layer 2, ReLU activation   
    hidden_layer2 <- sigmoid(hidden_layer1%*% W2 + matrix(rep(b2,N), nrow = N, byrow = T),SoftMax=TRUE)  
    hidden_layer2 <- matrix(hidden_layer2, nrow = N)   
    # class score   
    scores <- hidden_layer2%*%W3 + matrix(rep(b3,N), nrow = N, byrow = T)     
    # compute and normalize class probabilities    
    exp_scores <- exp(scores)   
    probs <- exp_scores / rowSums(exp_scores)    
    # compute the cross-entropy loss: sofmax and regularization   
   # corect_logprobs <- -log(probs)   
    data_loss <- sum((probs-Y)^2)   
    reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) +0.5*reg*sum(W3*W3)  
    loss <- data_loss + reg_loss   
    
    # compute the gradient on scores   
    dscores <- probs-Y   
    dscores <- dscores/N    
    
    # backpropate the gradient to the parameters   
    dW3 <- t(hidden_layer2)%*%dscores   
    db3 <- colSums(dscores)   
    
    # modify with momentum m
    dW3 <-dW3+(dW3)*m
    db3 <-db3+(db3)*m
    
    # next backprop into hidden layer 2
    dhidden2 <- dscores%*%t(W3)  
    # backprop the ReLU non-linearity  
    #  dhidden2[hidden_layer2 <= 0] <- 0   
    
    
    # backpropate the gradient to the parameters   
    dW2 <- t(hidden_layer1)%*%dhidden2   
    db2 <- colSums(dhidden2)   
    
    # modify with momentum m
    dW2 <-dW2+(dW2)*m
    db2 <-db2+(db2)*m
    
    # next backprop into hidden layer  
    dhidden1 <- dscores%*%t(W2)  
    # backprop the ReLU non-linearity  
    dhidden1[hidden_layer1 <= 0] <- 0   
    # finally into W,b   
    dW <- t(X)%*%dhidden1   
    db <- colSums(dhidden1) 
    
    # modify with momentum m
    dW <-dW+(dW)*m
    db <-db+(db)*m
    
    # add regularization gradient contribution 
    dW3 <- dW3 + reg *W3
    dW2 <- dW2 + reg *W2    
    dW <- dW + reg *W   
    
    # update parameter   
    W <- W-learningRate*dW    
    b <- b-learningRate*db   
    W2 <- W2-learningRate*dW2   
    b2 <- b2-learningRate*db2 
    W3 <- W3-learningRate*dW3   
    b3 <- b3-learningRate*db3   
    
    # check progress 
    if (i%%50 == 0 | i == niteration){ 
      values=0
      N_V=nrow(Xcv.proc)
      for(j in 1:length(ycv)){
        # values=values+(max((((matrix(Xcv.proc[j,],1,64)%*%W+b)%*%W2)+b2))-ycv[j]-1)^2
        hidden_layer1_V <- sigmoid(Xcv.proc%*% W + matrix(rep(b,N_V), nrow = N_V, byrow = T),SoftMax=TRUE)  
        hidden_layer1_V <- matrix(hidden_layer1_V, nrow = N_V) 
        hidden_layer2_V <- sigmoid(hidden_layer1_V%*% W2 + matrix(rep(b2,N_V), nrow = N_V, byrow = T),SoftMax=TRUE)  
        hidden_layer2_V <- matrix(hidden_layer2_V, nrow = N_V) 
        scores_V <- hidden_layer2_V%*%W3+ matrix(rep(b3,N_V), nrow = N_V, byrow = T)     
        # compute and normalize class probabilities    
        exp_scores_V <- exp(scores_V)   
        probs_V <- exp_scores_V / rowSums(exp_scores_V)    
        # compute the cross-entropy loss: sofmax and regularization   
        corect_logprobs_V <- -log(probs_V)   
        data_loss_V <- sum((probs_V-Y_V)^2)  
        reg_loss_V <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2) +0.5*reg*sum(W3*W3)
        loss_V <- data_loss_V + reg_loss_V  
        
      }
      validation_error <- c(validation_error, loss_V)
      #print(paste("iteration", i,': train loss', loss))#
      #print(paste("iteration", i,': validation error', validation_error[i%/%50+1]))#
      if(is.unsorted(rev(validation_error))==TRUE){
        break
      }  
      
    }
    
  }
  
  return(list(W, b, W2, b2,W3,b3))}

##predict function 
nnetPred <- function(X, para = list()){  
  W <- para[[1]]  
  b <- para[[2]] 
  W2 <- para[[3]] 
  b2 <- para[[4]]
  W3 <- para[[5]] 
  b3 <- para[[6]] 
  N <- nrow(X) 
  # hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))  
  #hidden_layer <- matrix(hidden_layer, nrow = N) 
  # scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)  
  
  hidden_layer1 <- sigmoid(X%*% W + matrix(rep(b,N), nrow = N, byrow = T),SoftMax=TRUE)  
  hidden_layer1 <- matrix(hidden_layer1, nrow = N)   
  # hidden layer 2, ReLU activation   
  hidden_layer2 <- sigmoid( hidden_layer1%*% W2 + matrix(rep(b2,N), nrow = N, byrow = T),SoftMax=TRUE)  
  hidden_layer2 <- matrix(hidden_layer2, nrow = N)   
  # class score   
  scores <- hidden_layer2%*%W3 + matrix(rep(b3,N), nrow = N, byrow = T) 
  predicted_class <- apply(scores, 1, which.max)   
  return(predicted_class)  }

##class accuracy
class_accuracy=function(x){
  class_accuracy=vector("numeric")
  for(i in 1:10){
    accuracy=diag(x)[i]/sum(x[,i])
    class_accuracy=c(class_accuracy,accuracy)
  }
  class_accuracy
}

##Performance of training set
ptm <- proc.time()
nnet.mnist <- nnet(X.proc, Y, Xcv.proc, Y_V,learningRate = 0.3, h = 10, m=0.05, niteration= 6000)
proc.time() - ptm
predicted_class <- nnetPred(X.proc, nnet.mnist)
print(paste('training set accuracy:', mean(predicted_class == (y+1))))
h=cbind(as.vector(y+1),as.vector(predicted_class)) 
confusionMatrix = table (h[,1], h[,2])
print(confusionMatrix)
class_accuracy(confusionMatrix)

##Performance of test  set
predicted_class_Test <- nnetPred(XT.proc, nnet.mnist)
print(paste('test set accuracy:',   mean(predicted_class_Test == (yT+1))))
h_t=cbind(as.vector(yT+1),as.vector(predicted_class_Test)) 
confusionMatrix_Test = table (h_t[,1], h_t[,2])
print(confusionMatrix_Test)
class_accuracy(confusionMatrix_Test)

