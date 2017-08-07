#-------------A simple Neural Network with 2 layers (fully connected)--------------- 
#First, I will train it to classify a set of 4-class 2D data and visualize the decision bounday. 
#Second, I am going to train my NN with the famous MNIST data (https://www.kaggle.com/c/digit-recognizer) and see its performance.
#https://www.r-bloggers.com/build-your-own-neural-network-classifier-in-r/

#fully connected layer: y = w1*x1+w2*x2+....+wn*xn, means all x contribute on y. So, linear transformation is the fully connected layer
#convolutional layer: input a image, maybe only x1,..x5 contribute on y.
#convolutional neural network: contains some convolutional layer.


library(ggplot2)
library(caret) 
N <- 200 # number of points per class
D <- 2 # dimensionality
K <- 4 # number of classes
X <- data.frame() # data matrix (each row = single example)
y <- data.frame() # class labels 
set.seed(308) 
for (j in (1:K)){r <- seq(0.05,1,length.out = N) # radius  
  t <- seq((j-1)*4.7,j*4.7, length.out = N) + rnorm(N, sd = 0.3) # theta  
  Xtemp <- data.frame(x =r*sin(t) , y = r*cos(t))
  ytemp <- data.frame(matrix(j, N, 1))  
  X <- rbind(X, Xtemp)  
  y <- rbind(y, ytemp)} 

data <- cbind(X,y)
colnames(data) <- c(colnames(X), 'label')


x_min <- min(X[,1])-0.2 
x_max <- max(X[,1])+0.2
y_min <- min(X[,2])-0.2
y_max <- max(X[,2])+0.2 
# lets visualize the data:
ggplot(data) + geom_point(aes(x=x, y=y, color = as.character(label)), size = 2) + 
  theme_bw(base_size = 15) +  xlim(x_min, x_max) + ylim(y_min, y_max) +  
  ggtitle('Spiral Data Visulization') +  coord_fixed(ratio = 0.8) +  
  theme(axis.ticks=element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        axis.text=element_blank(), axis.title=element_blank(), legend.position = 'none')



#====Neural network construction====
X <- as.matrix(X)
# for each example (each row in Y), the entry with index==label is 1 (and 0 otherwise).
Y <- matrix(0, N*K, K) 
for (i in 1:(N*K)){Y[i, y[i,]] <- 1}

# %*% dot product, * element wise product
nnet <- function(X, Y, step_size = 0.5, reg = 0.001, h = 10, niteration){  #h: 中间参数 the number of nero in the hidden layer, can be changed. （x：D dim, after the 1st layer, it change to h dim)
                                                                             #一般来说h越大越好，保存信息多 
  # number of examples 
  N <- nrow(X) 
  # number of classes  
  K <- ncol(Y) 
  # dimensionality  
  D <- ncol(X) 
  # initialize parameters randomly  
  W <- 0.01 * matrix(rnorm(D*h), nrow = D)  #fx = XW + b; X:N*D, W:D*h, b:1*h, fx:N*h=N*10
  b <- matrix(0, nrow = 1, ncol = h)   
  W2 <- 0.01 * matrix(rnorm(h*K), nrow = h)  # hx = fxW2 + b2; fx:N*h, W2:h*K, b2: 1*K, hx:N*K=N*10
  b2 <- matrix(0, nrow = 1, ncol = K)   
  # gradient descent loop to update weight and bias  
  for (i in 0:niteration){   
    # hidden layer, ReLU activation  # activation is only used on hidden layer
    hidden_layer <- pmax(0, X%*%W + matrix(rep(b,N), nrow = N, byrow = T))  # max(0, XW+b), pmax compare 0 with each element in the matrix by columns, returns a vector with length N*10.    
    hidden_layer <- matrix(hidden_layer, nrow = N)    
    # class score    
    scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)   # hx = fxW2 + b2  N*K
    # compute and normalize class probabilities    
    exp_scores <- exp(scores)    #N*K
    probs <- exp_scores / rowSums(exp_scores)  #softmax   N*K
    # compute the loss: sofmax and regularization    #softmax: 把一个 vector 归一到 0 到 1之间的数，bc we use cross entropy loss, which has logp. If we use other loss function, we may not need use softmax.
    corect_logprobs <- -log(probs)    #punish probs very small   N*K
    data_loss <- sum(corect_logprobs*Y)/N   # cross entropy,  corect_logprobs:N*K, Y:N*K, -->corect_logprobs*Y: N*K, element-wise multplication, result is a scalar
    reg_loss <- 0.5*reg*sum(W*W) + 0.5*reg*sum(W2*W2)    #Euclidean distance
    loss <- data_loss + reg_loss    
    # check progress    
    if (i%%1000 == 0 | i == niteration){
      print(paste("iteration", i,': loss', loss))}     
    # compute the gradient on scores    
    dscores <- probs-Y    # dim: N*K - N*K = N*K
    dscores <- dscores/N     
    # backpropate the gradient to the parameters    
    dW2 <- t(hidden_layer)%*%dscores  #h*N x N*K, -->h*K  # ddata_Loss/dw
    db2 <- colSums(dscores)    # N*1   
    # next backprop into hidden layer    
    dhidden <- dscores%*%t(W2)    # N*K x K*h --> N*h 
    # backprop the ReLU non-linearity    
    dhidden[hidden_layer <= 0] <- 0    
    # finally into W,b    
    dW <- t(X)%*%dhidden    
    db <- colSums(dhidden)     
    # add regularization gradient contribution    
    dW2 <- dW2 + reg *W2    #reg*W2: dreg_loss/dw
    dW <- dW + reg *W     
    # update parameter     
    W <- W-step_size*dW    
    b <- b-step_size*db    
    W2 <- W2-step_size*dW2    
    b2 <- b2-step_size*db2 }  
  return(list(W, b, W2, b2))}


#this model only use gradient descent, input all train data. 
#Not use stochastic gradient descent, which the gradient only use part of the train data.
#When dataset is large, we input a part of the data first,divide the whole train data into minibatches. 
#For each minibatch, stochastic gradient descent use the data in thi minibatch to calculate the gradient.
#After all minibatches are inputted, we say it is one epoch (one iteration)





#Prediction function and model training
nnetPred <- function(X, para = list()){  
  W <- para[[1]]  
  b <- para[[2]]  
  W2 <- para[[3]]  
  b2 <- para[[4]]   
  N <- nrow(X)  
  hidden_layer <- pmax(0, X%*% W + matrix(rep(b,N), nrow = N, byrow = T))   
  hidden_layer <- matrix(hidden_layer, nrow = N)  
  scores <- hidden_layer%*%W2 + matrix(rep(b2,N), nrow = N, byrow = T)   
  predicted_class <- apply(scores, 1, which.max)   
  return(predicted_class) } 

nnet.model <- nnet(X, Y, step_size = 0.4,reg = 0.0002, h=50, niteration = 6000)


predicted_class <- nnetPred(X, nnet.model)
print(paste('training accuracy:',mean(predicted_class == (y))))


#====Decision boundary====
# plot the resulting classifier
hs <- 0.01
grid <- as.matrix(expand.grid(seq(x_min, x_max, by = hs), seq(y_min, y_max, by =hs)))
Z <- nnetPred(grid, nnet.model) 

ggplot()+  geom_tile(aes(x = grid[,1],y = grid[,2],fill=as.character(Z)), alpha = 0.3, show.legend = F)+
  geom_point(data = data, aes(x=x, y=y, color = as.character(label)), size = 2) + theme_bw(base_size = 15) + 
  ggtitle('Neural Network Decision Boundary') + coord_fixed(ratio = 0.8) + 
  theme(axis.ticks=element_blank(),panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
        axis.text=element_blank(), axis.title=element_blank(), legend.position = 'none')





#===== MNIST data and preprocessing ======
# load Mnist data
train =  read.csv("~/Documents/UCSD-Stats/Neural Network/DigitsClassification/train.csv", header = TRUE, stringsAsFactors = F)
test = read.csv("~/Documents/UCSD-Stats/Neural Network/DigitsClassification/test.csv", header = TRUE, stringsAsFactors = F)

#=================
#The first column of the train and test datasets are labels
#Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. 
#Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, 
#with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
#To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, 
#where i and j are integers between 0 and 27, inclusive. 
#Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).
#==================


displayDigit <- function(X){  
  m <- matrix(unlist(X),nrow = 28,byrow = T)  
  m <- t(apply(m, 2, rev))  
  image(m,col=grey.colors(255))} 

displayDigit(train[18,-1])


#removing near zero variance columns and scaling by max(X).
nzv <- nearZeroVar(train)
nzv.nolabel <- nzv-1 
inTrain <- createDataPartition(y=train$label, p=0.7, list=F) 
training <- train[inTrain, ]
CV <- train[-inTrain, ]   # cross validation set

# data matrix (each row = single example)
X <- as.matrix(training[, -1]) 
# number of examples
N <- nrow(X) 
# class labels
y <- training[, 1] 
# number of classes
K <- length(unique(y)) 
# scale
X.proc <- X[, -nzv.nolabel]/max(X) 
# dimensionality 
D <- ncol(X.proc) 
# data matrix (each row = single example)
Xcv <- as.matrix(CV[, -1]) 
# class labels
ycv <- CV[, 1] 
# scale CV data 
Xcv.proc <- Xcv[, -nzv.nolabel]/max(X) 
# scale test data
Xtest <- as.matrix(test[, -1]) 
Xtest.proc <- Xtest[, -nzv.nolabel]/max(X)  #it is max(X), not max(Xtest), bc we don't know anything about test data
ytest <- test[,1]
  
  
# for each example (each row in Y), the entry with index==label is 1 (and 0 otherwise).
Y <- matrix(0, N, K) 
for (i in 1:N){Y[i, y[i]+1] <- 1}




#=====Model training and CV accuracy=====
nnet.mnist <- nnet(X.proc, Y, step_size = 0.3, reg = 0.0001, niteration = 3500)

predicted_class <- nnetPred(X.proc, nnet.mnist)
print(paste('training set accuracy:', mean(predicted_class == (y+1))))

predicted_class <- nnetPred(Xcv.proc, nnet.mnist)
print(paste('CV accuracy:', mean(predicted_class == (ycv+1))))

predicted_class <- nnetPred(Xtest.proc, nnet.mnist)
print(paste('test accuracy:', mean(predicted_class == (ytest+1))))


#====Prediction of a random image====
Xtest <- Xcv[sample(1:nrow(Xcv), 1), ]
Xtest.proc <- as.matrix(Xtest[-nzv.nolabel], nrow = 1)
predicted_test <- nnetPred(t(Xtest.proc), nnet.mnist)
print(paste('The predicted digit is:',predicted_test-1 ))

displayDigit(Xtest)














