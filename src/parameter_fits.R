library(lme4)
library(lmerTest)
library(ggplot2)
library(car)
library(caret)
library(latex2exp)
library(varhandle)
library(reshape2)

#####################################################################################################################
###################### Parameter estimates and model comparison   ###################################################
#####################################################################################################################
## By Tankred Saanum
#####################

#####################################################################################################################
###################################  Description and Contents #######################################################
#####################################################################################################################
##  This R script fits hyperparameters to the behavioral models we consider, and performs the model comparison as well
##  There's quite a bit of code in this script - here's how it's organized:
##  There are N sections, each containing 
##
##  1) In the first section I load the estimated differences in value produced by the different models
##     and save it in dataframes/arrays. These are used as predictors in mixed effects logistic regression models.
##     I also load the outcome variable "y", compute the mean rewards obtained by every subject, and some other
##     variables that are useful.
##
##  2) Here I write the model fitting functions that we use to compute best-fitting hyperparameters for the various 
##     models. There are quite a bit of functions here that were used for earlier models, but are not relevant for the
##     models used in the paper. There are also functions for computing BIC/AIC weights from log likelihoods, as well as some other
##     convenience functions for further analysis of the models.
##
##  3) In this section I fit the models and obtain the best hyperparameters, as well as their log likelihoods
##
##  4) In this section I look at differences in the contributions of the temporal/SR component vs the Euclidean component
##
##  5) Here I compute a weighting for the Euclidean predictor for every choice in the data set. This weight quantifies
##     the extent to which a choice is predicted by the Euclidean predictor over the temporal predictor.
##
##  6) In this section I compute AIC/BIC and R2, and plot these
##
##  7) Here I analyse the R2 a bit further, specifically, I look at how the models compare on trials
##     where subjects have observed the value for both options, and not, separately.
#####################################################################################################################
#####################################################################################################
###### SECTION 1:
###### Here I open the predictor csv files:
###### NB!!!! I specify the full path to the csv files, if you want to run this code yourself, you need to give the script your full path
###### or maybe you can figure out how to use the relative path (I can't seem to be able to do this with my R version)
###################################################
###################################################

### get hyperparameter data
sr <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\sr_results.csv"))
euc <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\euc_results.csv"))
temporal <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\sr_gp_results_v3_final.csv"))

compositional <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\comp_results_v3_final.csv"))


################
## optional models, only semi-relevant:
optimized_gp <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\optimized_results.csv"), header=FALSE)
optimized_gp <- optimized_gp[,-1]  ## remove index column

## SR Softmax
sr_softmax <-as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\softmax_sr.csv"), header=FALSE)
 
## mean tracker and random sr kernel
mean_tracker <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\mean_tracker.csv"), header=FALSE)
srgp_rand <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\SR_scrambled.csv"), header=TRUE)
####################

## get outcomes and some extra features for the random effects
behav_data <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\choice_data.csv"))

## unpack
subj <- unlist(behav_data["subj"])
trials <- unlist(behav_data["trial"])
trials_scaled <- scale(trials)  # standardize trials
y <- unlist(behav_data["decision"])
y <- y - 1  # subtract 1 so decision becomes a binary outcome variable 


### compute mean rewards obtained by subjects
rewards <- behav_data$chosen_value
subs <- unique(subj)
m.rewards <- rep(0, length(subs))
c <- 1
for (sub in subs){
  r <- rewards[which(subj==sub)]
  m.rewards[c] <- mean(r)
  c <- c+1
}

######################################################################
#### SECTION 2:
#### Define functions for fitting hyperparameters (through grid search)
######################################################################

## this function finds the best-fitting hyperparameter for each subject individually
find.minimum.individual <- function(data, y){
  subjects <- unique(subj)
  best.params <- rep(0, length(subjects))
  best.param.idx <- rep(0, length(subjects))
  c <- 1
  for (subj_id in subjects){
    subj_data <- data[which(subj== subj_id), ]
    response <- y[which(subj==subj_id)]
    output <- find.minimum.fixed(subj_data, response)
    best.params[c] <- output$best.param
    best.param.idx[c] <- output$idx
    c <- c+1
    
  }
  
  new.predictor <- rep(0, 100*length(subjects))
  c <- 0
  for (i in 1:length(subjects)){
    subj_id <- subjects[i]
    param.idx <- best.param.idx[i]
    best.fit.data <- data[which(subj== subj_id), param.idx]
    new.predictor[(c+1):(i*100)] <- best.fit.data
    c <- i * 100
  
    
  }
  list("predictor"= new.predictor, "params"= best.params)

}


### finds the individual minimum with an extra predictor which is constant
find.minimum.fixed.extra.predictor <- function(data, predictor2, y, extra.main = FALSE){
  names(data) <- substring(names(data),2)  # removes the annoying X in front of the parameter value
  param.values <- colnames(data)  # get column names
  loss <- rep(0, length(param.values))  # prepare loss array
  counter <- 1  # make a counter to index loss array
  if (is.vector(predictor2)){
    new.predictor2 <- scale(predictor2)
    is.vec <- TRUE
  } else {
    is.vec <- FALSE
  }
  
  for (i in 1:length(param.values)){
    param.col <- param.values[i]
    predictor <- data[, param.col]  # get the model predictor under the current parameter value
    predictor <- scale(predictor)
    
    if (!is.vec){
      new.predictor2 <- predictor2[, i]
      new.predictor2 <- scale(new.predictor2)
    }
    
    if (extra.main){
      model <- glm(y ~ -1  + new.predictor2 + predictor, family = "binomial")
    } else{
      model <- glm(y ~ -1  + predictor + new.predictor2, family = "binomial")
    }
      # create model
    loss[counter] <- -logLik(model)  # put log likelihood in at the appropriate index
    counter <- counter + 1  # increment counter
    
  }
  min.nll <- min(loss)  # get the best loss
  best.param <- param.values[which.min(loss)]  # get the parameter at the best loss
  
  list("min.nll" = min.nll, "best.param" = best.param, "idx" = which.min(loss))  # return these in a list
}


find.minimum.individual.extra.predictor <- function( data, extra_predictor, y, extra_main = FALSE){
  subjects <- unique(subj)
  best.params <- rep(0, length(subjects))
  best.param.idx <- rep(0, length(subjects))
  c <- 1
  for (subj_id in subjects){
    subj_data <- data[which(subj== subj_id), ]
    if (is.vector(extra_predictor)){
      predictor2 <- extra_predictor[which(subj==subj_id)]
    } else {
      predictor2 <- extra_predictor[which(subj==subj_id), ]
    }
    
    response <- y[which(subj==subj_id)]
    output <- find.minimum.fixed.extra.predictor(subj_data, predictor2 , response, extra.main = extra_main)
    best.params[c] <- output$best.param
    best.param.idx[c] <- output$idx
    c <- c+1
    
  }
  
  new.predictor <- rep(0, 100*length(subjects))
  c <- 0
  for (i in 1:length(subjects)){
    subj_id <- subjects[i]
    param.idx <- best.param.idx[i]
    best.fit.data <- data[which(subj== subj_id), param.idx]
    new.predictor[(c+1):(i*100)] <- best.fit.data
    c <- i * 100
    
    
  }
  list("predictor"= new.predictor, "params"= best.params)
  
}


# finds the individually best-fitting hyperparams for two predictors
find.minimum.individual.2d <- function(data1, data2, y){
  subjects <- unique(subj)
  best.params1 <- rep(0, length(subjects))
  best.params2 <- rep(0, length(subjects))
  best.param1.idx <- rep(0, length(subjects))
  best.param2.idx <- rep(0, length(subjects))
  weights <- rep(0, length(subjects))
  beta1 <- rep(0, length(subjects))
  beta2 <- rep(0, length(subjects))
  c <- 1
  for (subj_id in subjects){
    subj_data1 <- data1[which(subj== subj_id), ]
    subj_data2 <- data2[which(subj== subj_id), ]
    response <- y[which(subj==subj_id)]
    output <- find.minimum.2d(subj_data1, subj_data2, response)
    
    
    
    best.params1[c] <- output$best.param1
    best.params2[c] <- output$best.param2
    best.param1.idx[c] <- output$idx1
    best.param2.idx[c] <- output$idx2
    weights[c] <- output$weight
    beta1[c] <- output$beta1
    beta2[c] <- output$beta2
    c <- c+1
    
  }
  
  new.predictor1 <- rep(0, 100*length(subjects))  # 100 observations per participant
  new.predictor2 <- rep(0, 100*length(subjects))
  c <- 0
  for (i in 1:length(subjects)){
    subj_id <- subjects[i]
    param.idx1 <- best.param1.idx[i]
    param.idx2 <- best.param2.idx[i]
    best.fit.data1 <- data1[which(subj== subj_id), param.idx1]
    best.fit.data2 <- data2[which(subj== subj_id), param.idx2]
    new.predictor1[(c+1):(i*100)] <- best.fit.data1
    new.predictor2[(c+1):(i*100)] <- best.fit.data2
    c <- i * 100
    
    
  }
  new.df <- data.frame(new.predictor1, new.predictor2)
  list("predictors"= new.df, "params1"= best.params1, "params2" = best.params2, "weights" = weights, "beta1" = beta1, "beta2" = beta2)
  
}


########
## finds the 
######

find.minimum.2d <- function(data1, data2, y){
  names(data1) <- substring(names(data1),2)  
  names(data2) <- substring(names(data2),2)  
  param.values1 <- colnames(data1)  # get column names
  param.values2 <- colnames(data2)
  loss <- rep(0, (length(param.values1)*length(param.values2)))  # prepare loss array

  
  w_beta <- rep(0, (length(param.values1)*length(param.values2)))
  coef1 <- rep(0, (length(param.values1)*length(param.values2)))
  coef2 <- rep(0, (length(param.values1)*length(param.values2)))
  
  beta
  counter <- 1
  param.indices <- expand.grid(seq(1, length(param.values1)), seq(1, length(param.values2)))
  
  for (param.col1 in param.values1){
    for (param.col2 in param.values2){
      predictor1 <- data1[, param.col1]
      predictor1 <- scale(predictor1)
      
      predictor2 <- data2[, param.col2]
      predictor2 <- scale(predictor2)
      model <- glm(y ~ -1 + predictor1 + predictor2, family = "binomial")
      beta1 <- coef(model)[[1]]
      beta2 <- coef(model)[[2]]
      b1w <- abs(beta1)/(abs(beta1) + abs(beta2))  # use the absolute value to get the contribution of each beta
      # we only save the first beta weight as the second one can be computed as (1 - b1w)
      w_beta[counter] <- b1w
      coef1[counter] <- beta1
      coef2[counter] <- beta2
      
      loss[counter] <- -logLik(model)
      counter <- counter + 1
      
    }
  }
  
  min.nll <- min(loss)  # get the best loss
  min.idx <- which.min(loss)
  best.param.indices <- param.indices[min.idx, ]
  #print(best.param.indices)
  best.param.idx1 <- best.param.indices[[1]]
  best.param.idx2 <- best.param.indices[[2]]
  #print(best.param.idx2)
  
  best.param1 <- param.values1[best.param.idx1]
  best.param2 <- param.values2[best.param.idx2]
  #print(best.param2)
  W <- w_beta[min.idx]
  optimal.beta1 <- coef1[min.idx]
  optimal.beta2 <- coef2[min.idx]
  
  list("min.nll" = min.nll, "best.param1" = best.param1, "best.param2" = best.param2,
       "idx1" = best.param.idx1, "idx2" = best.param.idx2, "weight" = W, "beta1" = optimal.beta1, "beta2" = optimal.beta2)  # return these in a list
  
  
}


##### find minimum among a set of 1d paramater values (only fixed effects, for per participant parameter estimates)
find.minimum.fixed <- function(data, y){
  names(data) <- substring(names(data),2)  # removes the annoying X in front of the parameter value
  param.values <- colnames(data)  # get column names
  loss <- rep(0, length(param.values))  # prepare loss array
  counter <- 1  # make a counter to index loss array
  for (param.col in param.values){
    predictor <- data[, param.col]  # get the model predictor under the current parameter value
    predictor <- scale(predictor)
    model <- glm(y ~ -1 + predictor, family = "binomial")  # create model
    loss[counter] <- -logLik(model)  # put aic in at the appropriate index
    counter <- counter + 1  # increment counter
    
  }
  min.nll <- min(loss)  # get the best loss
  best.param <- param.values[which.min(loss)]  # get the parameter at the best loss
  
  list("min.nll" = min.nll, "best.param" = best.param, "idx" = which.min(loss))  # return these in a list
}

##############
### This is the function we actually use to find the population level hyperparams, and to compute the log likelihoods
### It's quite simple: it takes a matrix of predictors (each reflecting a particular hyperparameter for the model) and
### gives the log likelihood of the best fitting hyperparam, as well as the column index of that predictor in the matrix
####
find.minimum <- function(data, y){
  names(data) <- substring(names(data),2)  # removes the annoying X in front of the parameter value
  param.values <- colnames(data)  # get column names
  loss <- rep(0, length(param.values))  # prepare loss array
  counter <- 1  # make a counter to index loss array
  for (param.col in param.values){
    predictor <- data[, param.col]  # get the model predictor under the current parameter value
    predictor <- scale(predictor)
    model <- glmer(y ~ -1 + predictor + (-1 + predictor|subj), family = "binomial")  # create model
    loss[counter] <- -logLik(model)  # put nLL in at the appropriate index
    counter <- counter + 1  # increment counter
    
  }
  min.nll <- min(loss)  # get the best loss
  best.param <- param.values[which.min(loss)]  # get the parameter at the best loss

  list("min.nll" = min.nll, "best.param" = best.param, "idx" = which.min(loss), "dist" =loss)  # return these in a list
}


# reshapes an array of 1x4800 into 48x100
reshape.mixed.effects <- function(effects){
  mat <- matrix(effects, nrow = 48, ncol=100, byrow=TRUE)
  mat
}


##### functions for computing aic/bic and aic/bic weights

compute.aic <- function(nll, p){
  ll <- -nll
  (p*2) - (2*ll)
}


compute.bic <- function(nll, p, n){
  
  ll <- -nll
  (p*log(n)) - (2*ll)
  }

compute.weights <- function(scores){
  
  min.score <- min(scores)  # get the minimum aic/bic
  diff.scores <-rep(0, length(scores))
  c <- 1
  
  # compute differences
  for (score in scores){
    diff <- score - min.score
    diff.scores[c] <- diff
    c <- c+1
  }
  weights <-rep(0, length(scores))
  c <- 1
  
  # compute AIC/BIC weights
  for (diff in diff.scores){
    w <- exp(-0.5 * diff) / sum(exp(-0.5*diff.scores))
    weights[c] <- w
    c <- c+1
  }
  
  list("diff" = diff.scores, "weights" = weights)
  
}

#########################################################################
######### SECTION 3:
###############################################################
### apply "find.minimum" to all datasets and look at the output
#############################################################


####
#SR
####
successor_rep.output <- find.minimum(sr, y)
successor_rep.output$min.nll
successor_rep.output$best.param
##
## Temporal map
successor.output <- find.minimum(temporal, y)
successor.output$min.nll
successor.output$idx
successor.output$best.param


sr.p <- 3
##
## sr softmax: This model is not included in the analysis, but could still be interesting. 
## It's a SR-GP model in which the SR is updated trial by trial according to the softmax of the estimated rewards of each monster
sr.softmax.output <- find.minimum(sr_softmax, y)
sr.softmax.output$min.nll


###
#Euclidean
###
euc.output <- find.minimum(euc, y)
euc.output$min.nll
euc.output$best.param
eu.idx <- euc.output$idx

euc.p <- 3

###


###
# Compositional kernel
##


comp.output <- find.minimum(compositional, y)
comp.output$min.nll
comp.output$best.param
comp.output$idx
comp.output$dist


param.grid.comp <- expand.grid(seq(0.01, 0.7, length.out = 20), seq(0.1, 4, length.out = 20))
param.grid.comp[comp.output$idx,]

comp.p <- 4  # number of free parameters


## compute the %correct in predicting choices for the compositional model

best.comp.model <- glmer(y~-1 + compositional[, comp.output$idx] + (-1 + compositional[, comp.output$idx]|subj), family="binomial")

logLik(best.comp.model)


best.model.preds <- predict(best.comp.model, type="response")
best.model.preds[best.model.preds < 0.5] <- 0
best.model.preds[best.model.preds >= 0.5] <- 1

error <- sum(abs(best.model.preds - y))
num.correct <- length(y) - error
percent.correct.predictions <- num.correct / length(y)
percent.correct.predictions



##### Mean tracker
MT <- scale(mean_tracker)
MT_mod <- glmer(y~-1 + MT +(-1 + MT|subj), family="binomial")
MT_nll <- -logLik(MT_mod)

MT_nll

##############
### here are some alternative models, and some control models
################
### 
## optimized gp
##

op_gp_mod <- glmer(y~ -1 + optimized_gp + (-1 + optimized_gp|subj), family="binomial")
logLik(op_gp_mod)




# #### compositional with random SR-kernel
# 
# comp_rand <- scale(comp_rand_sr)
# comp_rand_mod <- glmer(y~-1 + comp_rand + (-1 + comp_rand|subj), family="binomial")
# logLik(comp_rand_mod)
# summary(comp_rand_mod)

### compositional with identity matrix
comp_id <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\compositional_identity.csv"), header=FALSE)

comp_id <- scale(comp_id)
comp_id_mod <- glmer(y~-1 + comp_id + (-1 + comp_id|subj), family="binomial")
logLik(comp_id_mod)


comp.id.df <- data.frame(names = c("Spatio-temporal", "Spatial + diagonal"),
                         vals = c(comp.output$min.nll- comp.output$min.nll, -logLik(comp_id_mod) - comp.output$min.nll))

id.colors= c("#2f9893", "#c20a2b" )
ggplot(comp.id.df, aes(x=names, y = vals)) +
  geom_bar(stat="identity", fill =id.colors , width=0.4) +
  scale_x_discrete() +
  theme_minimal() + theme(aspect.ratio = 1.2, axis.text.x=element_text(size=15), axis.text.y=element_text(size=18),
                          axis.title=element_text(size=25))+
  xlab("") + ylab(TeX("$\\Delta$ AIC"))


##########
## sr gp paired with random temporal kernels
srgp_rand <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\SR_scrambled.csv"), header=TRUE)

num.tau <- ncol(srgp_rand)
mixed.log.liks <- rep(0, num.tau)
for (i in 1:num.tau){
  sr.mixed.predictor <- scale(srgp_rand[, i])
  sr.mixed.mod <- glmer(y~-1 + sr.mixed.predictor + (-1 + sr.mixed.predictor|subj), family="binomial")
  mixed.log.liks[i] <- -logLik(sr.mixed.mod)
  
}

tau.avg <- rep(0, 10)
tau.sd <- rep(0, 10)
tau.X <- rep(0, 100)
tau.vals <- seq(from = 0.05, to = 3, length.out = 10)

for (i in 1:10){
  tau.avg[i] <- mean(mixed.log.liks[i:(i+9)])
  tau.sd[i] <- sd(mixed.log.liks[i:(i+9)])
  
  tau.X[(1+((i-1)*10)):(i*10)] <- tau.vals[i]
  
}


tau.df <- data.frame(X= tau.X, y=mixed.log.liks, unscrambled = rep(successor.output$min.nll, length(tau.X)))
tau.box <- data.frame(X = as.factor(tau.X), y =mixed.log.liks, unscrambled = rep(successor.output$min.nll, length(tau.X)))
tau.df <- data.frame(X = tau.vals, y=tau.avg, se=tau.sd, unscrambled = rep(successor.output$min.nll, length(tau.avg)))
cor.test(mixed.log.liks, tau.X)




ggplot(tau.df, aes(x=X, y=y)) +
  geom_line(aes( colour = "Shuffled"), size=1.2) +
  xlab(label=TeX("$\\tau$")) + 
  ylab("Loss") +
  geom_errorbar(aes(ymin=y-se, ymax=y+se), width=0.2, size=1.3, colour = "#c20a2b") +
  geom_line(aes(x=X, y = unscrambled, group = 1, colour = "True"), linetype="dashed",size=1.8) +
  #geom_point(size=5, colour="#c20a2b") +
  theme_minimal()+
  theme(axis.text.y=element_text(size=20), axis.text.x=element_text(size=20),axis.title.x=element_text(size=35,face="bold"), 
        axis.title.y=element_text(size=25), legend.text=element_text(size=25)) +
  scale_x_continuous(limit=c(0,  3.1), breaks=c(0, 0.5, 1, 1.5, 2, 2.5, 3) ) +
  scale_y_continuous(limit=c(2470, 2530)) + 
  scale_colour_manual("", 
                      breaks = c("Shuffled", "True"),
                      values = c( "#c20a2b", "#4ab4da"))



## plot the control model
# ggplot(tau.box, aes(x=X, y=y)) +
#   #geom_point(size=2) + 
#   geom_boxplot() + 
#   geom_line(aes(x=X, y = unscrambled, group = 1), linetype="dashed",size=1.2, colour = "#2b3852") + 
#   theme_minimal()+
#   theme(axis.text=element_text(size=12),axis.title.x=element_text(size=35,face="bold"), axis.title.y=element_text(size=15,face="bold")) + 
#   #geom_smooth(method="lm")+
#   scale_x_continuous(limit=c(0,  3.1), breaks = ) +
#   xlab(label=TeX("$\\tau$")) + 
#   ylab(label="Loss")+
#   annotate("text", x=0.16, y=2480, label=TeX("r = -0.27", output='character'), parse=TRUE, size=6)+
#   annotate("text", x=0.2, y=2474, label=TeX("p = 0.007", output='character'), parse=TRUE, size=6)

    
    





#####################################################################
######################################################################
#### Section 3.1
#### Here we save the best fitting predictors in a seperate file and use them to
#### do leave-one-out cross validation 



LOO.CV <- function(dat, preds){
  y <- dat$y
  N <- length(y)
  subj <- dat$subj
  all.idx <- seq(N)
  loss <- matrix(nrow=N, ncol=length(colnames(preds)))
  progress <- 1
  steps <- N*length(colnames(preds))
  for (i in seq(1:N)){
    train.idx <- all.idx[-i]
    test.idx <- all.idx[i]
    
    for (j in 1:length(colnames(preds))){
      name <- colnames(preds)[j]
      train.data <- data.frame(y=y[train.idx], X = preds[train.idx, name], id = subj[train.idx])
      test.data <- data.frame(y=y[test.idx], X = preds[test.idx, name], id = subj[test.idx])
      mod <- glmer(y~ -1 + X + (-1 + X|id), family = "binomial", data=train.data)
      
      loo.prediction <- predict(mod, newdata=test.data,  type="response")
      outcome <- y[test.idx]
      loglikelihood <- log((loo.prediction * outcome)+((1-loo.prediction) * (1-outcome)))
      loss[i, j] <- loglikelihood
      
      cat("\r",progress/steps)
      progress <- progress + 1
      
    }
  }
  loss <- as.data.frame(loss)
  colnames(loss) <- colnames(preds)
  loss
}

model.analysis <- data.frame(y=y, subj = subj)

final.predictors <- data.frame(euc = euc[, euc.output$idx],
                              temp = temporal[, successor.output$idx], comp = compositional[, comp.output$idx], mt = MT[, 1])



## beware, this takes a couple of hours to run
losses <- LOO.CV(model.analysis, final.predictors)

losses.df <- data.frame(losses, id = subj)
write.csv(losses.df, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\loocv_choice.csv", row.names=F)




### plot the outputs of the bayesian model analysis

choice.freq <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_freq_choice.csv"))


pxp.colors <- c("#e4a021", "#4ab4da", "#2f9893", "#5d6264")
models <- c('Spatial', 'Temporal', 'Spatio-temporal', 'None')
mean <- choice.freq$mean
mean
se <- sqrt(choice.freq$var)
best.idx <- which.max(choice.freq$xp)
exc <- rep(0, 4)
exc[best.idx] <- choice.freq$xp[best.idx]
#exc.line <- c(exc[best.idx], exc[best.idx], exc[best.idx], exc[best.idx])
exc.line <- c(1, 1, 1, 1)
exc.pointer <- c(best.idx, best.idx, best.idx, best.idx)
exc.pointer.y <- c(1.001, 0.95, 0.9, 0.9)

pxp.df <- data.frame(models, mean, se, exc, exc.line, exc.pointer, exc.pointer.y)
limits <- aes(ymax = mean + se, ymin=mean - se)
exc[best.idx]




ggplot(pxp.df, aes(x = models, y=mean)) +
  geom_bar(stat="identity", fill=pxp.colors, width=0.65) +
  geom_errorbar(limits, position="dodge", width=0.2, size=1.3) +
  geom_line(aes(x=models, y = exc.line, group = 1), linetype="solid",size=1.2, colour = "#2b3852") +
  geom_line(aes(x=exc.pointer, y = exc.pointer.y, group = 1), linetype="solid",size=1.2, colour = "#2b3852") +
  #coord_cartesian(ylim=c(0, 1.2)) +
  
  # #geom_text(aes(x=3.3, y=0.93, label=("$P_{exc}=0.995$")), size=5) +
  # #  geom_abline(intercept = comp.exc, slope = 0, linetype="dashed", size=2.5) +
  theme_bw()+
  theme(aspect.ratio = 1.2, panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border=element_blank(), 
        axis.text.x=element_text(size=20, angle=40, vjust = 1.25, hjust=1.08, colour = pxp.colors),
        axis.text.y=element_text(size=18),
        axis.title = element_text(size=18),
        plot.title = element_text(size = 18, face = "bold"))+
  
  scale_x_discrete(limits = models) +
  scale_y_continuous(expand = c(0, 0), limit=c(-0.1,  1.2), breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
  xlab("") + 
  ylab("Model frequency") +
  annotate("text", x=2, y=1.1, label=TeX("$P_{exc} >   0.999", output='character'), parse=TRUE, size=6) +
  annotate(x=0, xend=0, y=0, yend=1., colour="darkgray", lwd=1.75, geom="segment")+
  annotate(x=0, xend=3.5, y=0, yend=0, colour="darkgray", lwd=1., geom="segment")









######################################################################
#####################################################################
#### Section 4
### Here I look at relative contribution of euclidean vs SR-GP/temporal models
### I create two composite models, using either the euclidean or the temporal 
### predictors as a main predictor, and a decorrelated version of the other as
### a second predictor, as well as interaction terms with trials.
### I then analyse how well each subject is described by any of the predictors on average
### I also perform VIF and condition index analyses of these models.
#################################################################################
###################################################################################

pred.srgp <- scale(temporal[, successor.output$idx])  # get the best fitting temporal predictor
pred.euc <- scale(euc[, euc.output$idx])  # get the best euclidean predictor

trials_scaled <- c(trials_scaled)  # make trials_scaled a vector

## fit the two composite models, using trial interactions
composite1 <- glmer(y~ -1 + pred.srgp + (pred.euc - pred.srgp) + (pred.srgp*trials_scaled) + ((pred.euc - pred.srgp)* trials_scaled) + (-1 + pred.srgp + (pred.euc - pred.srgp)+ (pred.srgp*trials_scaled)|subj) , family="binomial")
composite2 <- glmer(y~ -1 + pred.euc + (pred.srgp - pred.euc) +(pred.euc* trials_scaled) + ((pred.srgp - pred.euc)* trials_scaled) + (-1 + pred.euc + (pred.srgp - pred.euc) + (pred.euc*trials_scaled)|subj) , family="binomial")
##


###############################################################
## now make models with intercept so we can perform VIF analysis
composite2_intercept <- glmer(y~ pred.euc + (pred.srgp - pred.euc) +(pred.euc* trials_scaled) + ((pred.srgp - pred.euc)* trials_scaled) + (-1 + pred.euc + (pred.srgp - pred.euc) + (pred.euc*trials_scaled)|subj) , family="binomial")
composite1_intercept <- glmer(y~  pred.srgp + (pred.euc - pred.srgp) + (pred.srgp*trials_scaled) + ((pred.euc - pred.srgp)* trials_scaled) + (-1 + pred.srgp + (pred.euc - pred.srgp)+ (pred.srgp*trials_scaled)|subj) , family="binomial")

### condition index analysis
var1 <- pred.srgp
var2 <- (pred.euc - pred.srgp)
var3 <- scale(pred.srgp*trials_scaled)
var4 <- scale((pred.euc - pred.srgp)*trials)


data = data.frame(var1, var2, var3, var4)
mat <- cor(data)
e <-eigen(mat)
ev <- e$values
emax <- max(ev)
cn <- sqrt(emax/ev)
max(cn)

## vif

vif(composite1_intercept)
vif(composite2_intercept)

#######################################################
## back to analysing the composite models without intercepts

summary(composite1)
summary(composite2)


## get effects
sr.effect <- coef(composite1)$subj
euc.effect <- coef(composite2)$subj
# accumulate the effects over the models
sr.subjective.effect <- abs(sr.effect$pred.srgp) + abs(euc.effect$pred.srgp)
euc.subjective.effect <- abs(euc.effect$pred.euc) + abs(sr.effect$pred.euc)

plot(euc.subjective.effect, abs(euc.trial.effect))
plot(sr.subjective.effect, euc.subjective.effect)
cor.test(sr.subjective.effect, euc.subjective.effect)

# compute weight as proportion of total effect
w.euc <- euc.subjective.effect / (sr.subjective.effect + euc.subjective.effect)

## compute the trial interaction effects for the models in which this predictor had a random effect
sr.trial.effect <- abs(sr.effect$`pred.srgp:trials_scaled`) 
euc.trial.effect <- abs(euc.effect$`pred.euc:trials_scaled`)


plot(euc.trial.effect, sr.trial.effect)
cor.test(euc.trial.effect, sr.trial.effect)
## compute the weight the same way as with the main effect
trialw <- euc.trial.effect /(euc.trial.effect + sr.trial.effect)



cor.test(m.rewards, w.euc)
cor.test(w.euc, trialw)

create.hist <- function(df, name, lab, title, breaks=1){
  p <- ggplot(df, aes(x=name)) +geom_histogram(binwidth =breaks) + xlab(lab) + theme(axis.text=element_text(size=20),
                                                                      axis.title=element_text(size=14,face="bold"))  + ggtitle(title) + theme_bw()
  p
}

create.corr.plot <- function(df, name1, name2, lab1, lab2){
  p <-  ggplot(df, aes(x=name1, y=name2)) + geom_point(size=2) + 
    xlab(lab1) + ylab(lab2) + theme(axis.text=element_text(size=12),
                                                  axis.title=element_text(size=14,face="bold")) + geom_smooth(method="lm") #+ theme_bw()
  
  p
}

model.diagnostics.df <- data.frame(sr.subjective.effect, euc.subjective.effect, w.euc, m.rewards, sr.trial.effect, euc.trial.effect, trialw)
## plot for weights and reward
create.corr.plot(model.diagnostics.df, w.euc, m.rewards, "Weight Euclidean", "Mean rewards")
## correlation of effects
create.corr.plot(model.diagnostics.df, sr.subjective.effect, euc.subjective.effect, "SR GP effects", "Euclidean effects")
## correlation of effect in late vs early
create.corr.plot(model.diagnostics.df, sr.trial.effect, euc.trial.effect, "SR GP effects over trials", "Euclidean effects over trials")
## correlation of weight of effect, late vs. early, and performance
create.corr.plot(model.diagnostics.df, trialw, m.rewards, "Weighting of Late Euclidean influence", "Mean reward")
cor.test(sr.subjective.effect, euc.subjective.effect)
cor.test(w.euc, m.rewards)
## effect per subject
plot(sr.subjective.effect, euc.subjective.effect)

ggplot(model.diagnostics.df, aes(x=w.euc)) + theme_bw() +geom_histogram(binwidth = 0.02, color="black") + xlab("Weight Euclidean")+ ggtitle("") + theme(axis.text=element_text(size=20),
                                                                                                                                           axis.title=element_text(size=14,face="bold")) + scale_x_continuous(breaks = round(seq(0, 1, by = 0.1),1))

cor.test(trialw, m.rewards)

## write these things into csv
write.csv(model.diagnostics.df, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\effects_and_weights.csv")


###############################################
#### SECTION 5
############### 
## this is where I compute the contribution of each predictor for each trial:
################

## this convenience function just expands a vector of shape 1x48 to a vector of
## 1 x 4800. This is used to reconstruct predictions of y using single predictors
## and estimated coefficients

expand.mixed.effects <- function(ME){
  expanded <- rep(0, length(ME)*100)
  c <- 1
  for (effect in ME){
    
    expanded[c:(c+99)] <- effect
    c <- c + 100
  }
  expanded
}

## a function for reconstructing a prediction made by either of the composite models in the last section

reconstruct.y <- function(pred1, b1, pred2, b2, pred3, b3, pred4, b4, pred5, b5){
  b1 <- expand.mixed.effects(b1)
  b2 <- expand.mixed.effects(b2)
  b3 <- expand.mixed.effects(b3)
  b4 <- expand.mixed.effects(b4)
  b5 <- expand.mixed.effects(b5)
  
  term1 <- pred1 * b1
  term2 <- pred2 * b2
  term3 <- pred3 * b3
  term4 <- pred4 * b4
  term5 <- pred5 * b5
  y.prime <- term1 + term2 + term3 + term4 + term5
  y.prime
}
## a simple logistic function with k=1 (and thus omitted)
logistic <- function(X){
  1/ (1 + exp(-(X)))
}

reconstruct.abs <- function(pred1, b1, pred2, b2, pred3, b3, pred4, b4, pred5, b5){
  ## this one is to measure how much a particular prediciton is influenced by the various predictors
  b1 <- expand.mixed.effects(b1)
  b2 <- expand.mixed.effects(b2)
  b3 <- expand.mixed.effects(b3)
  b4 <- expand.mixed.effects(b4)
  b5 <- expand.mixed.effects(b5)
  
  term1 <- (pred1 * b1)
  term2 <- (pred2 * b2)
  term3 <- (pred3 * b3)
  term4 <- (pred4 * b4)
  term5 <- (pred5 * b5)
  
  y.prime <- term1 + term2 +term3  + term4 + term5
  
  
  pred1 <- logistic(term4)
  pred2 <- logistic(term5)

  
  eff1 <- ((y - pred1)**2)
  eff2 <- ((y - pred2)**2)
  


  list("eff1"=eff1, "eff2"=eff2, "y.prime"=pred1)
}



composite.preds1 <- predict(composite1, re.form= NULL)
composite.preds2 <- predict(composite2, re.form=NULL)

composite.betas1 <- coef(composite1)
composite.betas2 <- coef(composite2)
## random effects c1
c1.sr <- composite.betas1$subj$pred.srgp
c1.euc <- composite.betas1$subj$pred.euc
c1.t <- composite.betas1$subj$trials_scaled
c1.sr_t <- composite.betas1$subj$`pred.srgp:trials_scaled`
c1.euc_t <- composite.betas1$subj$`pred.euc:trials_scaled`
## random effects c2
c2.sr <- composite.betas2$subj$pred.srgp
c2.euc <- composite.betas2$subj$pred.euc
c2.t <- composite.betas2$subj$trials_scaled
c2.sr_t <- composite.betas2$subj$`pred.srgp:trials_scaled`
c2.euc_t <- composite.betas2$subj$`pred.euc:trials_scaled`


# get design matrices used in model and make predictors out of them
design.matrix.1 <- getME(composite1, name="X")
design.matrix.2 <- getME(composite2, name="X")
dX1 <- sapply(design.matrix.1, as.matrix)
dX1 <- matrix(dX1, nrow=4800, ncol=5)
dX2 <- sapply(design.matrix.2, as.matrix)
dX2 <- matrix(dX2, nrow=4800, ncol=5)


## here I just make sure the reconstructed predictions match those of the model:
y.prime.1 <- reconstruct.y(dX1[, 1], c1.sr, dX1[, 2], c1.euc, dX1[, 3], c1.t, dX1[, 4], c1.sr_t, dX1[, 5], c1.euc_t)
plot(composite.preds1, y.prime.1)

y.prime.2 <- reconstruct.y(dX2[, 1], c2.euc, dX2[, 2], c2.sr, dX2[, 3], c2.t, dX2[, 4], c2.euc_t, dX2[, 5], c2.sr_t)
plot(composite.preds2, y.prime.2)

## indeed they do

## now I estimate contributions
c1.effects.per.trial <- reconstruct.abs(dX1[, 1], c1.sr, dX1[, 2], c1.euc, dX1[, 3], c1.t, dX1[, 4], c1.sr_t, dX1[, 5], c1.euc_t)
sr.effect.per.trial1 <- c1.effects.per.trial$eff1 
euc.effect.per.trial1 <- c1.effects.per.trial$eff2

## note that for the second model, the euclidean is the main predictor, and therefore under the "eff1" name
# rather than "eff2"
c2.effects.per.trial <- reconstruct.abs(dX2[, 1], c2.euc, dX2[, 2], c2.sr, dX2[, 3], c2.t, dX2[, 4], c2.euc_t, dX2[, 5], c2.sr_t)
euc.effect.per.trial2 <- c2.effects.per.trial$eff1 
sr.effect.per.trial2 <- c2.effects.per.trial$eff2

## take the average contribution over the two models
sr.per.trial <- (sr.effect.per.trial1+ sr.effect.per.trial2)/2
euc.per.trial <- (euc.effect.per.trial2 + euc.effect.per.trial2)/2

# get the relative errors of the Euclidean predictor (later to be used as weight)
error_w <- euc.per.trial/ (euc.per.trial + sr.per.trial)
plot(sr.per.trial, euc.per.trial)


### reshape into matrix, and make it 1 - error_w, so larger weight means larger contribution, instead of the other way around
error.matrix <- matrix(1-error_w, nrow=48, ncol=100, byrow=T)
mean.error.w <- colMeans(error.matrix)

## plot and correlate with trials
plot(trial_array, mean.error.w )
cor.test(trial_array, mean.error.w)


## now we want to reshape it to 48 x 100 so we can look at how these things progress over trials on average
sr.per.trial.mat <- matrix(sr.per.trial, nrow = 48, ncol=100, byrow=T)
euc.per.trial.mat <- matrix(euc.per.trial, nrow=48, ncol=100, byrow = T)
avg.mat <- (euc.per.trial.mat) / (euc.per.trial.mat + sr.per.trial.mat)
plot(trial_array, colMeans(avg.mat))
# uncomment to save csv
write.csv(error.matrix, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\tBt_euc_w.csv", row.names=F)


# compare high performing vs low performing
sorted.idx <- order(m.rewards)
top.ten <- sorted.idx[39:48]
bottom.ten <- sorted.idx[1:10]

HP.euc.mat <- error.matrix[top.ten, ]
LP.euc.mat <- error.matrix[bottom.ten, ]

HP.euc.w <- colMeans(HP.euc.mat)
LP.euc.w <- colMeans(LP.euc.mat)

avg.euc <- colMeans(euc.per.trial.mat)
avg.sr <- colMeans(sr.per.trial.mat)

plot(trial_array, 1/ avg.sr)


average.euclidean <- avg.euc/(avg.sr + avg.euc)
per.trial.df <- data.frame(avg.euc, avg.sr, trial_array, average.euclidean, LP.euc.w, mean.error.w, HP.euc.w)
write.csv(per.trial.df,"C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\per_trial_df.csv" , row.names = F)
ggplot(per.trial.df, aes(x=trial_array, y=mean.error.w)) + geom_point() + geom_smooth() + ylim(0, 1) +
  theme_bw() + xlab("Trials") + ylab("Average euclidean weight") +theme(axis.text=element_text(size=12),
                                                                       axis.title=element_text(size=14,face="bold"))


ggplot(per.trial.df, aes(x=trial_array, y=mean.error.w)) + geom_point() + geom_smooth() + ylim(0, 1) +
  theme_bw() + xlab("Trials") + ylab("Average euclidean weight") +theme(axis.text=element_text(size=12),
                                                                        axis.title=element_text(size=14,face="bold"))
ggplot(per.trial.df, aes(x=trial_array, y=LP.euc.w)) + geom_point() + geom_smooth() + theme_bw() + ylim(0, 1) +
  xlab("Trials") + ylab("Average euclidean weight (low performing)") +theme(axis.text=element_text(size=12),
                                                             axis.title=element_text(size=14,face="bold"))

ggplot(per.trial.df, aes(x=trial_array, y=HP.euc.w)) +geom_point()+ geom_smooth() + theme_bw() + ylim(0, 1)+
  xlab("Trials") + ylab("Average euclidean weight (high performing)") +theme(axis.text=element_text(size=12),
                                                                            axis.title=element_text(size=14,face="bold"))


cor.test(mean.error.w, trial_array)

## this is a convenience function for computing how well a subjects weight over trials correlate with trial
compute.w.corrs <- function(matrix){
  corrs <- rep(0, nrow(matrix))
  for (i in 1:nrow(matrix)){
    delta_w <- matrix[i, ]
    c <- cor(trial_array, delta_w)
    corrs[i] <- c
  }
  corrs
}


## basically the same as the one above, just computing OLS betas instead of correlation
compute.w.betas <- function(matrix){
  betas <- rep(0, nrow(matrix))
  for (i in 1:nrow(matrix)){
    delta_w <- matrix[i, ]
    
    covariance <- cov(trial_array, delta_w)
    beta <- covariance/(var(delta_w))
    betas[i] <- beta
  }
  betas
}

hist(compute.w.corrs(error.matrix))
corrs.array <- compute.w.corrs(error.matrix)
plot(corrs.array, m.rewards)
cor.test(corrs.array, m.rewards)
betas <- compute.w.betas(error.matrix)


trial.analysis.df <- data.frame(corrs.array, m.rewards, w.euc, trialw, betas)
ggplot(trial.analysis.df, aes(x=corrs.array, y=m.rewards)) + geom_point() + geom_smooth(method="lm") + theme_bw() +
  xlab("Rho: Trial and Eucidean weight") + ylab("Mean rewards") +theme(axis.text=element_text(size=12),
                                                                       axis.title=element_text(size=14,face="bold"))

ggplot(trial.analysis.df, aes(x=betas, y=m.rewards)) + geom_point() + geom_smooth(method="lm") + theme_bw() +
  xlab("Beta: Trial and Eucidean weight") + ylab("Mean rewards") +theme(axis.text=element_text(size=12),
                                                                        axis.title=element_text(size=14,face="bold"))



## this function fits a logistic function to all weight timeseries for subjects, and outputs the fitted curves, as well
## as the estimated inflection points. There's also an excluded trial parameter which is an array of trials excluded from the fitting
test.dat <- data.frame(X=seq(1:100))

compute.logistic.fit <- function(matrix, excluded.trials){
  trials.X <- seq(1:100)
  trials.X <- trials.X[!(trials.X %in% excluded.trials)]
  inflection.points <- rep(0, nrow(matrix))
  coefs <- rep(0, nrow(matrix))
  full.trials <- seq(1:100)
  
  #predictions <- matrix(nrow=nrow(matrix), ncol = length(trials.X))
  predictions <- matrix(nrow=nrow(matrix), ncol = length(full.trials))
  for (i in 1:nrow(matrix)){
    delta_w <- matrix[i, trials.X]
    dat <- data.frame(y = delta_w, X = trials.X)
    new.dat <- data.frame(X = full.trials)
    fit <- glm(y ~ X, family=binomial(link=logit), data = dat)
    k <- coef(fit)[2]
    ## predict for full trial set
    preds <- predict(fit, newdata=new.dat, type="response")
    
    #preds <- predict(fit, type="response")
    predictions[i, ] <- preds
    b <- (1/preds[1])- 1
    infl <- log(b)/k
    inflection.points[i] <- infl
    coefs[i] <- k
    

  }
  list("curve" = predictions, "inflection.point" = inflection.points, "coefs" = coefs)
}

excluded.trials <- c(1, 2, 11, 12, 51)  ## these trials the weights are 0.5 for trivial reasons
indices <- seq(1:100)
trials.X <- indices[!(indices %in% excluded.trials)]
logistic.fits <- compute.logistic.fit(error.matrix, excluded.trials)
plot(logistic.fits$inflection.point, m.rewards)
## save curves to csv
write.csv(logistic.fits$curve, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\log_curves_weights.csv", row.names=F)
change_statistics <- data.frame(logistic.fits$coefs, abs(logistic.fits$inflection.point))
header <- c("slopes", "inflection_points")
names(change_statistics) <- header
write.csv(change_statistics, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\logistic_slopes.csv", row.names = F)

hist(logistic.fits$inflection.point, breaks = 20)
neg.idx <- which(logistic.fits$inflection.point < 0)
## get the indices of subjects with positive logistic functions
pos.idx <- which(logistic.fits$inflection.point > 0)

inflection.df <- data.frame(R= m.rewards[pos.idx], inflection=logistic.fits$inflection.point[pos.idx])
### plot inflection points
ggplot(inflection.df, aes(x=inflection, y=R)) + geom_point() + geom_smooth(method="lm") + theme_bw() +
  xlab("Estimated inflection point (trials)") + ylab("Mean reward") +theme(axis.text=element_text(size=12),
                                                                             axis.title=element_text(size=14,face="bold"))

cor.test(logistic.fits$inflection.point[pos.idx], m.rewards[pos.idx])



######################################################
#### SECTION 5.1
#####################################################
#### As a sanity check, we want to see if we get similar plots when we make the mean tracker and the euclidean
#### predictors compete in predicting choices. Ideally, we'd like to see a high effect for the euclidean compared to mean
#### tracker throughout the experiment


### MT and Euclidean together
## exactly the same set up as with the EUC Vs. SR models, only that I swap the SR-predictor for the mean tracker (MT) predictor.

mt_euc_mod1 <- glmer(y~-1 + MT + ( pred.euc- MT )+ (MT* trials_scaled) + (( pred.euc- MT )*trials_scaled) +(-1 + MT + ( pred.euc- MT ) + (MT* trials_scaled) + (( pred.euc- MT )*trials_scaled)|subj), family="binomial")
mt_euc_mod2 <- glmer(y~-1 + pred.euc + (MT - pred.euc)+ (pred.euc* trials_scaled) + ((MT - pred.euc)*trials_scaled) +(-1 + pred.euc + (MT - pred.euc) + (pred.euc* trials_scaled) + ((MT - pred.euc)*trials_scaled)|subj), family="binomial")

logLik(mt_euc_mod1)
logLik(mt_euc_mod2)
summary(mt_euc_mod1)
summary(mt_euc_mod2)

MTEUC.preds1 <- predict(mt_euc_mod1, re.form= NULL)
MTEUC.preds2 <- predict(mt_euc_mod2, re.form=NULL)
#corr.preds <- predict(composite1_intercept, re.form=NULL)

MTEUC.betas1 <- coef(mt_euc_mod1)
MTEUC.betas2 <- coef(mt_euc_mod2)


### check if we can reproduce the logistic functions with the correlated model
corr.betas <- coef(composite1_intercept)
corr.euc <- corr.betas$subj$pred.euc
corr.sr <- corr.betas$subj$pred.srgp
corr.t <- corr.betas$subj$trials_scaled
corr.euc_t <- corr.betas$subj$`pred.euc:trials_scaled`
corr.sr_t <- corr.betas$subj$`pred.srgp:trials_scaled`
M <- getME(composite1_intercept, name="X")
M <- sapply(M, as.matrix)
M <- matrix(M, nrow=4800, ncol=5)
y.prime.1 <- reconstruct.y(M[, 1], corr.sr, M[, 2], corr.euc, M[, 3], corr.t, M[, 4], corr.sr_t, M[, 5], corr.euc_t)
plot(corr.preds, y.prime.1)

eff.per.trial <- reconstruct.abs(M[, 1], corr.sr, M[, 2], corr.euc, M[, 3], corr.t, M[, 4], corr.sr_t, M[, 5], corr.euc_t)
sr.eff <- eff.per.trial$eff1
euc.eff <- eff.per.trial$eff2
euc.effect.corr <- euc.eff / (euc.eff + sr.eff)
euc.effect.corr <- matrix(1- euc.effect.corr, nrow = 48, 100, byrow=T)
mu.euc.corr <- colMeans(euc.effect.corr)
plot(trial_array, mu.euc.corr)
## indeed we see a similar logistic shape, but slightly weaker than with the uncorrelated technique
## this is nice!


## random effects c1
c1.mt <- MTEUC.betas1$subj$MT
c1.euc <- MTEUC.betas1$subj$pred.euc
c1.t <- MTEUC.betas1$subj$trials_scaled
c1.mt_t <- MTEUC.betas1$subj$`MT:trials_scaled`
c1.euc_t <- MTEUC.betas1$subj$`pred.euc:trials_scaled`
## random effects c2
c2.mt <- MTEUC.betas2$subj$MT
c2.euc <- MTEUC.betas2$subj$pred.euc
c2.t <- MTEUC.betas2$subj$trials_scaled
c2.mt_t <- MTEUC.betas2$subj$`MT:trials_scaled`
c2.euc_t <- MTEUC.betas2$subj$`pred.euc:trials_scaled`


# get design matrices used in model and make predictors out of them
design.matrix.1 <- getME(mt_euc_mod1, name="X")
design.matrix.2 <- getME(mt_euc_mod2, name="X")
dX1 <- sapply(design.matrix.1, as.matrix)
dX1 <- matrix(dX1, nrow=4800, ncol=5)
dX2 <- sapply(design.matrix.2, as.matrix)
dX2 <- matrix(dX2, nrow=4800, ncol=5)


## here I just make sure the reconstructed predictions match those of the model:
y.prime.1 <- reconstruct.y(dX1[, 1], c1.mt, dX1[, 2], c1.euc, dX1[, 3], c1.t, dX1[, 4], c1.mt_t, dX1[, 5], c1.euc_t)
plot(MTEUC.preds1, y.prime.1)

y.prime.2 <- reconstruct.y(dX2[, 1], c2.euc, dX2[, 2], c2.mt, dX2[, 3], c2.t, dX2[, 4], c2.euc_t, dX2[, 5], c2.mt_t)
plot(MTEUC.preds2, y.prime.2)

## indeed they do

## now I estimate contributions
c1.effects.per.trial <- reconstruct.abs(dX1[, 1], c1.mt, dX1[, 2], c1.euc, dX1[, 3], c1.t, dX1[, 4], c1.mt_t, dX1[, 5], c1.euc_t)
MT.effect.per.trial1 <- c1.effects.per.trial$eff1 
euc.effect.per.trial1 <- c1.effects.per.trial$eff2

## note that for the second model, the euclidean is the main predictor, and therefore under the "eff1" name
# rather than "eff2"
c2.effects.per.trial <- reconstruct.abs(dX2[, 1], c2.euc, dX2[, 2], c2.mt, dX2[, 3], c2.t, dX2[, 4], c2.euc_t, dX2[, 5], c2.mt_t)
euc.effect.per.trial2 <- c2.effects.per.trial$eff1 
MT.effect.per.trial2 <- c2.effects.per.trial$eff2

## take the average contribution over the two models
MT.per.trial <- (MT.effect.per.trial1+ MT.effect.per.trial2)/2
euc.per.trial <- (euc.effect.per.trial2 + euc.effect.per.trial2)/2

# get the relative errors of the Euclidean predictor (later to be used as weight)
error_w <- euc.per.trial/ (euc.per.trial + MT.per.trial)
plot(MT.per.trial, euc.per.trial)

avg.euc.per.trial.MT <- matrix(euc.per.trial, nrow=48, ncol=100, byrow = T)
avg.MT.per.trial.MT <- matrix(MT.per.trial, nrow=48, ncol=100, byrow = T)

exclude = c(1, 2, 10, 11)
plot(trial_array[-exclude], 1/colMeans(avg.euc.per.trial.MT)[-exclude])
plot(trial_array[-exclude], 1/colMeans(avg.MT.per.trial.MT)[-exclude])
plot(trial_array[-exclude], 1/avg.sr[-exclude])


avg.MT.pred <- colMeans(matrix(pred.euc, nrow=48, ncol=100, byrow=T))
avg.sr.pred <- colMeans(matrix(pred.srgp, nrow = 48, ncol=100, byrow=T))
cor.test(avg.sr.pred, avg.MT.pred)
plot(trial_array, avg.sr.pred)
plot(trial_array, avg.MT.pred)
cor.test(avg.sr.pred[-seq(80,100)], avg.MT.pred[-seq(80, 100)])
plot(avg.sr.pred[-seq(80,100)], avg.MT.pred[-seq(80, 100)])

### reshape into matrix, and make it 1 - error_w, so larger weight means larger contribution, instead of the other way around
error.matrix <- matrix(1-error_w, nrow=48, ncol=100, byrow=T)
mean.error.euc.MT <- colMeans(error.matrix)

## plot and correlate with trials
plot(trial_array, mean.error.euc.MT)
## we see the same shape, suggesting that the late euclidean influence isnt very well
## explained by the mean tracker
cor.test(trial_array, mean.error.euc.MT)





###############################################
####### SECTION 6
###############################################
###  Here I compute the AIC and BIC differences
### as well as mc fadden's pseudo R2 statistic

random.loss <- log(0.5) * 4800  # the log probability of 0.5^4800 = log(0.5)*4800

r2 <- function(nll, random.loss){
  ll <- -nll
  1 - (ll/random.loss)
}

## compute R2 statistic

comp_R2 <- r2(comp.output$min.nll, random.loss)
euc_R2 <- r2(euc.output$min.nll, random.loss)
temporal_R2 <- r2(successor.output$min.nll, random.loss)
MT_R2 <- r2(-logLik(MT_mod), random.loss)

r2.stats <- c(euc_R2, temporal_R2,comp_R2, MT_R2)

###
##### Here we compute the aic and bic weights with the best parameters. 

nlls <- c(euc.output$min.nll, successor.output$min.nll, comp.output$min.nll, MT_nll)

params <- c(3, 3, 4, 2)


N <- 4800
aics <- compute.aic(nlls, params)
bics <- compute.bic(nlls, params, N)
aic.stats <- compute.weights(aics)
bic.stats <- compute.weights(bics)



sprintf("%.8f", aic.stats$weights)
sprintf("%.8f", bic.stats$weights)
sprintf("%.8f", aic.stats$diff)
sprintf("%.8f", bic.stats$diff)

names <- c("Spatial", "Temporal", "Spatio-temporal", "None")
colors <- c("#e4a021", "#4ab4da", "#2f9893", "#5d6264")
comparison.df <- data.frame(aic.stats$weights, aic.stats$diff, bic.stats$weights, bic.stats$diff,  names, r2.stats)


### full comparison


ggplot(data=comparison.df, aes(x=names, y=aic.stats$diff)) +
  geom_bar(stat="identity", fill=colors, width=0.65)+
  theme_minimal() + theme(aspect.ratio = 1.2, axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))  + scale_x_discrete(limits = names) +
  xlab("") + ylab(TeX("$\\Delta$ AIC")) +theme(axis.text.x=element_text(size=20, angle=40, vjust = 1.18, hjust=1.1, colour = colors), axis.text.y=element_text(size=18), axis.title=element_text(size=25), 
                                           plot.title = element_text(size = 18, face = "bold"))




###############################################
####### SECTION 7
###############################################
###  Here I compare model performance agains an idealized model, which knows the true
###  value of all monsters. I compare this model agains all other models on trials
###  where subjects have indeed seen the rewards for both monsters in the choice set
###  and do not need to employ any generalization. We compare these models in terms
###  of mc fadden's pseudo R2 statistic

true.diff <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\true_diff.csv"))
observed.both <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\observed_both.csv"))
observed.idx <- which(observed.both == 1) ## get trials where both monsters were observed

### Ideal model
true.diff <- scale(true.diff)  # scale predictor
ideal.mod <- glmer(y ~ -1 + true.diff + (-1 + true.diff|subj), family="binomial")
ideal.mod.obs <- glmer(y[observed.idx] ~ -1 + true.diff[observed.idx] + (-1 + true.diff[observed.idx]|subj[observed.idx]), family="binomial")
#ideal.mod.obs <- glm(y[observed.idx] ~ -1 + true.diff[observed.idx] , family="binomial")
logLik(ideal.mod.obs)

rand.ll <- log(0.5)*length(observed.idx)
ideal.r2 <- 1 - (logLik(ideal.mod.obs) / rand.ll)
ideal.as.line <- c(ideal.r2,ideal.r2, ideal.r2, ideal.r2)


comparison.df <- data.frame(aic.stats$weights, aic.stats$diff, bic.stats$weights, bic.stats$diff,  names, r2.stats, ideal.as.line)
comparison.df$names <- unfactor(comparison.df$names)



# p<-ggplot(data=comparison.df) +
#   geom_bar(aes(x=names, y=r2.stats), width=0.65, stat="identity", fill=colors)+
#   geom_abline(aes(colour = "Ideal", intercept = as.numeric(ideal.r2), slope = 0), 
#               linetype="dashed", size=1.5, colour = "#e65a77")+
#   theme_minimal() +  ylab("R2")+theme(aspect.ratio = 1.2, axis.text.x=element_text(size=20, angle=40, vjust = 1.1, hjust=0.95, colour = colors),legend.text = element_text(size=15, face="bold"),
#                                       axis.title = element_text(size=13, face="bold"), axis.text=element_text(size=15)) + ggtitle("") + xlab("") +
#   ylim(0,.4)+ scale_x_discrete(limits = names) + scale_color_manual(name="", values = c( 'Ideal' = '#e65a77'))
# p


ggplot(data=comparison.df) +
  geom_bar(aes(x=names, y=r2.stats), width=0.65, stat="identity", fill=colors)+
  geom_abline(aes(intercept = as.numeric(ideal.r2), slope = 0, colour = "Upper\n bound"), 
              linetype="dashed", size=1.5)+
  theme_minimal() +  ylab(TeX("McFadden's $R^2$"))+theme(aspect.ratio = 1.2, axis.text.y = element_text(size=20), axis.text.x=element_text(size=20, angle=40, vjust = 1.1, hjust=0.95, colour = colors),legend.text = element_text(size=20),
                                      axis.title = element_text(size=20), axis.text=element_text(size=20)) + ggtitle("") + xlab("") +
  ylim(0,.4)+ scale_x_discrete(limits = names) + scale_color_manual(name="", values = c( 'Upper\n bound' = '#e65a77'))


#plot
#######################

### check how well the other models do on this data set
### get model predictors

comp.obs <- compositional[observed.idx, comp.output$idx]  # the compositional predictor with best fitting parameters, at observed indices
euc.obs <- euc[observed.idx, euc.output$idx]  # the same for euc
sr.obs <- temporal[observed.idx, successor.output$idx]  # the same for temporal
mt.obs <- mean_tracker[observed.idx,]  # for the mean tracker
observed.ll <- log(0.5) * length(observed.idx)

### estimate models
#compositional
comp.obs.mod <-  glmer(y[observed.idx] ~ -1 + comp.obs + (-1 + comp.obs|subj[observed.idx]), family="binomial")
comp.obs.mod.fixed <-  glm(y[observed.idx] ~ -1 + comp.obs , family="binomial")
logLik(comp.obs.mod)

#euclidean
euc.obs.mod <-  glmer(y[observed.idx] ~ -1 + euc.obs + (-1 + euc.obs|subj[observed.idx]), family="binomial")
euc.obs.mod.fixed <-  glm(y[observed.idx] ~ -1 + euc.obs , family="binomial")
logLik(euc.obs.mod)

# temporal
sr.obs.mod <-  glmer(y[observed.idx] ~ -1 + sr.obs + (-1 + sr.obs|subj[observed.idx]), family="binomial")
sr.obs.mod.fixed <-  glm(y[observed.idx] ~ -1 + sr.obs , family="binomial")
logLik(sr.obs.mod)

# mean tracker
mt.obs.mod <-  glmer(y[observed.idx] ~ -1 + mt.obs + (-1 + mt.obs|subj[observed.idx]), family="binomial")
mt.obs.mod.fixed <-  glm(y[observed.idx] ~ -1 + mt.obs , family="binomial")
logLik(mt.obs.mod)


comp.O.r2 <- r2(-logLik(comp.obs.mod), observed.ll)
euc.O.r2 <- r2(-logLik(euc.obs.mod), observed.ll)
temporal.O.r2 <- r2(-logLik(sr.obs.mod), observed.ll)
mt.O.r2 <- r2(-logLik(mt.obs.mod), observed.ll)

r2.observed <- c(euc.O.r2, temporal.O.r2, comp.O.r2, mt.O.r2 )

############### 
## now let's see how these models perform on trials where one of the monsters' values werent observed previously.

unobserved.idx <- which(observed.both == 0)
unobserved.ll <- log(0.5) * length(unobserved.idx)

comp.obs <- compositional[unobserved.idx, comp.output$idx]  # the compositional predictor with best fitting parameters, at observed indices
euc.obs <- euc[unobserved.idx, euc.output$idx]  # the same for euc
sr.obs <- temporal[unobserved.idx, successor.output$idx]  # the same for temporal
mt.obs <- mean_tracker[unobserved.idx,]  # for the mean tracker

### estimate models
#compositional
comp.obs.mod <-  glmer(y[unobserved.idx] ~ -1 + comp.obs + (-1 + comp.obs|subj[unobserved.idx]), family="binomial")
comp.obs.mod.fixed <-  glm(y[unobserved.idx] ~ -1 + comp.obs , family="binomial")
logLik(comp.obs.mod)

#euclidean
euc.obs.mod <-  glmer(y[unobserved.idx] ~ -1 + euc.obs + (-1 + euc.obs|subj[unobserved.idx]), family="binomial")
euc.obs.mod.fixed <-  glm(y[unobserved.idx] ~ -1 + euc.obs , family="binomial")
logLik(euc.obs.mod)

# temporal
sr.obs.mod <-  glmer(y[unobserved.idx] ~ -1 + sr.obs + (-1 + sr.obs|subj[unobserved.idx]), family="binomial")
sr.obs.mod.fixed <-  glm(y[unobserved.idx] ~ -1 + sr.obs , family="binomial")
logLik(sr.obs.mod)

# mean tracker
mt.obs.mod <-  glmer(y[unobserved.idx] ~ -1 + mt.obs + (-1 + mt.obs|subj[unobserved.idx]), family="binomial")
mt.obs.mod.fixed <-  glm(y[unobserved.idx] ~ -1 + mt.obs , family="binomial")
logLik(mt.obs.mod)


comp.UO.r2 <- r2(-logLik(comp.obs.mod), unobserved.ll)
euc.UO.r2 <- r2(-logLik(euc.obs.mod), unobserved.ll)
temporal.UO.r2 <- r2(-logLik(sr.obs.mod), unobserved.ll)
mt.UO.r2 <- r2(-logLik(mt.obs.mod), unobserved.ll)

r2.unobserved <- c(euc.UO.r2, temporal.UO.r2, comp.UO.r2, mt.UO.r2 )

############ make some plots ############
comparison.df <- data.frame(aic.stats$weights, aic.stats$diff, bic.stats$weights, bic.stats$diff,  names, r2.stats, ideal.as.line, r2U = r2.unobserved, r2O = r2.observed)
comparison.df$names <- unfactor(comparison.df$names)

## observed 
p<-ggplot(data=comparison.df) +
  geom_bar(aes(x=names, y=r2O), width=0.65, stat="identity", fill=colors)+
  geom_abline(intercept = as.numeric(ideal.r2), slope = 0, 
              linetype="dashed", size=1.5, colour = "#e65a77")+
  theme_minimal() +  ylab("R2")+theme(aspect.ratio = 1.2, axis.text.x=element_text(size=20, angle=40, vjust = 1.1, hjust=0.95, colour = colors),legend.text = element_text(size=15, face="bold"),
                                      axis.title = element_text(size=13, face="bold"), axis.text=element_text(size=15)) + ggtitle("") + xlab("") +
  ylim(0,.4)+ scale_x_discrete(limits = names) #+ scale_color_manual(name="", values = c( 'Ideal' = '#e65a77'))
p


### unobserved

p<-ggplot(data=comparison.df) +
  geom_bar(aes(x=names, y=r2U), width=0.65, stat="identity", fill=colors)+
  geom_abline(intercept = as.numeric(ideal.r2), slope = 0, 
              linetype="dashed", size=1.5, colour = "#e65a77")+
  theme_minimal() +  ylab("R2")+theme(aspect.ratio = 1.2, axis.text.x=element_text(size=20, angle=40, vjust = 1.1, hjust=0.95, colour = colors),legend.text = element_text(size=15, face="bold"),
                                      axis.title = element_text(size=13, face="bold"), axis.text=element_text(size=15)) + ggtitle("") + xlab("") +
  ylim(0,.4)+ scale_x_discrete(limits = names) #+ scale_color_manual(name="", values = c( 'Ideal' = '#e65a77'))
p




#####################################
################
## Analysis of value ratings
################
#####################################
val.ratings <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\value_ratings.csv"))
val.ratings<- val.ratings[complete.cases(val.ratings), ]


inf.objects1 <- c(5, 11)
inf.objects2 <- c(18, 24)
inf.objects2.alt <- c(18-12, 24-12)


val.ratings.1 <- val.ratings[, 1:12]
val.ratings.2 <- val.ratings[, 13:24]

val.ratings.noninf1 <- val.ratings.1[, -inf.objects1]
val.ratings.noninf2 <- val.ratings.2[, -inf.objects2.alt]



val.ratings.inf1 <- val.ratings.1[, inf.objects1]
val.ratings.inf2 <- val.ratings.2[, inf.objects2.alt]

val.id <- rep(0, 48*24)  # id vector for all observations
val.id.inf <- rep(0, 48*4) # id vector for inference object observations exclusively
val.id.noninf <- rep(0, 48*20) # id vector for all non-inference object observations exclusively
counter1 <- 1
counter2 <- 1
counter3 <- 1
for (i in seq(1:48)){
  
  val.id[counter1:(counter1+23)] <- i
  val.id.inf[counter2:(counter2+3)] <- i
  val.id.noninf[counter3:(counter3+19)] <- i
  counter1 <- counter1 + 24
  counter2 <- counter2 + 4
  counter3 <- counter3 + 20
}

## now we need to merge the predictions across contexts
euclidean.ratings1 <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\fmri\\predictions\\euc_final_predictions1.csv"))
euclidean.ratings2 <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\fmri\\predictions\\euc_final_predictions2.csv"))


euclidean.ratings1 <- euclidean.ratings1[, 2:13]
euclidean.ratings2 <- euclidean.ratings2[, 2:13]

euclidean.ratings.inf1 <- euclidean.ratings1[, inf.objects1]  
euclidean.ratings.inf2 <- euclidean.ratings2[, inf.objects2.alt]

euclidean.ratings.noninf1 <- euclidean.ratings1[, -inf.objects1]  
euclidean.ratings.noninf2 <- euclidean.ratings2[, -inf.objects2.alt]


temporal.ratings1 <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\fmri\\predictions\\temporal_final_predictions1.csv"))
temporal.ratings2 <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\fmri\\predictions\\temporal_final_predictions2.csv"))


temporal.ratings1 <- temporal.ratings1[, 2:13]
temporal.ratings2 <- temporal.ratings2[, 2:13]

temporal.ratings.inf1 <- temporal.ratings1[, inf.objects1]
temporal.ratings.inf2 <- temporal.ratings2[, inf.objects2.alt]

temporal.ratings.noninf1 <- temporal.ratings1[, -inf.objects1]
temporal.ratings.noninf2 <- temporal.ratings2[, -inf.objects2.alt]



compositional.ratings1 <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\fmri\\predictions\\comp_final_predictions1.csv"))
compositional.ratings2 <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\fmri\\predictions\\comp_final_predictions2.csv"))


compositional.ratings1 <- compositional.ratings1[, 2:13]
compositional.ratings2 <- compositional.ratings2[, 2:13]

compositional.ratings.inf1 <- compositional.ratings1[, inf.objects1]
compositional.ratings.inf2 <- compositional.ratings2[, inf.objects2.alt]

compositional.ratings.noninf1 <- compositional.ratings1[, -inf.objects1]
compositional.ratings.noninf2 <- compositional.ratings2[, -inf.objects2.alt]


MT.ratings1 <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\fmri\\predictions\\MT_final_predictions1.csv"))
MT.ratings2 <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\fmri\\predictions\\MT_final_predictions2.csv"))


MT.ratings1 <- MT.ratings1[, 2:13]
MT.ratings2 <- MT.ratings2[, 2:13]


# MT.ratings.inf1 <- MT.ratings1[, inf.objects1]
# MT.ratings.inf2 <- MT.ratings2[, inf.objects2.alt]


MT.ratings.noninf1 <- MT.ratings1[, -inf.objects1]
MT.ratings.noninf2 <- MT.ratings2[, -inf.objects2.alt]

MT.means1 <- rowMeans(MT.ratings.noninf1)
MT.means2 <- rowMeans(MT.ratings.noninf2)
MT.ratings.inf1 <- matrix(nrow = 48, ncol = 2)
MT.ratings.inf2 <- matrix(nrow = 48, ncol = 2)
for (i in 1:48){
  MT.ratings.inf1[i, ] <- c(MT.means1[i], MT.means1[i])
  
  MT.ratings.inf2[i, ] <- c(MT.means2[i], MT.means2[i])

}


euc.val.pred <- c(as.vector(t(euclidean.ratings1)), as.vector(t(euclidean.ratings2)))
euc.val.pred.inf <- c(as.vector(t(euclidean.ratings.inf1)), as.vector(t(euclidean.ratings.inf2)))
euc.val.pred.noninf <- c(as.vector(t(euclidean.ratings.noninf1)), as.vector(t(euclidean.ratings.noninf2)))

temporal.val.pred <- c(as.vector(t(temporal.ratings1)), as.vector(t(temporal.ratings2)))
temporal.val.pred.inf <- c(as.vector(t(temporal.ratings.inf1)), as.vector(t(temporal.ratings.inf2)))
temporal.val.pred.noninf <- c(as.vector(t(temporal.ratings.noninf1)), as.vector(t(temporal.ratings.noninf2)))


comp.val.pred <- c(as.vector(t(compositional.ratings1)), as.vector(t(compositional.ratings2)))
comp.val.pred.inf <- c(as.vector(t(compositional.ratings.inf1)), as.vector(t(compositional.ratings.inf2)))
comp.val.pred.noninf <- c(as.vector(t(compositional.ratings.noninf1)), as.vector(t(compositional.ratings.noninf2)))

mt.val.pred <-  c(as.vector(t(MT.ratings1)), as.vector(t(MT.ratings2)))
mt.val.pred.inf <- c(as.vector(t(MT.ratings.inf1)), as.vector(t(MT.ratings.inf2)))
mt.val.pred.noninf <- c(as.vector(t(MT.ratings.noninf1)), as.vector(t(MT.ratings.noninf2)))


y.val <- c(as.vector(t(val.ratings.1)), as.vector(t(val.ratings.2)))
y.val.inf <- c(as.vector(t(val.ratings.inf1)), as.vector(t(val.ratings.inf2)))
y.val.noninf <- c(as.vector(t(val.ratings.noninf1)), as.vector(t(val.ratings.noninf2)))

## now we want to make dataframes containing predictions for inference objects only, and another for everything but the inference objects


val.df <- data.frame(y = scale(y.val), euclidean = scale(euc.val.pred), temporal = scale(temporal.val.pred), mt = scale(mt.val.pred), comp = scale(comp.val.pred), id = val.id)
val.csv <- data.frame(y = scale(y.val), euclidean = scale(euc.val.pred), temporal = scale(temporal.val.pred), comp = scale(comp.val.pred), id = val.id)

val.df.inf <- data.frame(y = scale(y.val.inf), euclidean = scale(euc.val.pred.inf), temporal = scale(temporal.val.pred.inf),comp =scale(comp.val.pred.inf), mt = scale(mt.val.pred.inf), id = val.id.inf)
val.df.noninf <- data.frame(y = scale(y.val.noninf), euclidean = scale(euc.val.pred.noninf), temporal = scale(temporal.val.pred.noninf),comp =scale(comp.val.pred.noninf), mt = scale(mt.val.pred.noninf), id = val.id.noninf)


######################################################################
########################################################################
#### Now let's estimate our models. We begin with the non-inference objects

euclidean.val.mod.noninf <- lmer(y ~euclidean + (-1 + euclidean | id), data = val.df.noninf)

temporal.val.mod.noninf <- lmer(y ~temporal + (-1 + temporal | id), data = val.df.noninf)

comp.val.mod.noninf <- lmer(y ~comp + (-1 + comp | id), data = val.df.noninf)

mt.val.mod.noninf <- lmer(y ~mt + (-1 + mt | id), data = val.df.noninf)


AIC(euclidean.val.mod.noninf)
AIC(temporal.val.mod.noninf)
AIC(comp.val.mod.noninf)
AIC(mt.val.mod.noninf)



###
minaic.noninf <- min(c(AIC(euclidean.val.mod.noninf),
                     AIC(temporal.val.mod.noninf),
                     AIC(comp.val.mod.noninf),
                     AIC(mt.val.mod.noninf)))

val.aic.diffs.noninf <- c(AIC(euclidean.val.mod.noninf) - minaic.noninf, AIC(temporal.val.mod.noninf) - minaic.noninf,
                         AIC(comp.val.mod.noninf) - minaic.noninf, AIC(mt.val.mod.noninf) - minaic.noninf)





val.names <- c("Spatial", "Temporal", "Spatio-temporal", "None")
val.colors <- c("#e4a021", "#4ab4da", "#2f9893", "#5d6264")
val.comparison.df <- data.frame(colors = val.colors, names= val.names, aic= val.aic.diffs.noninf)

p<-ggplot(data=val.comparison.df, aes(x=names, y=aic)) +
  geom_bar(stat="identity", fill=val.colors, width=0.65) +
  theme_minimal() +
  xlab("") +
  ylab("AIC difference") +
  scale_x_discrete(limits = val.names) +
  theme(aspect.ratio = 1.2, axis.text.x = element_text(size=20, angle=40, vjust = 1.18, hjust=1.1, colour = val.colors), 
        axis.text.y=element_text(size=18), axis.title=element_text(size=18)) #+
p





###############################################
###################
### now for the inference objects




euclidean.val.mod <- lmer(y ~ euclidean + (-1 + euclidean | id), data = val.df.inf)
#summary(euclidean.val.mod)
temporal.val.mod <- lmer(y ~ temporal + (-1 + temporal | id), data = val.df.inf)
#summary(temporal.val.mod)

comp.val.mod <- lmer(y ~ comp + (-1 + comp | id), data = val.df.inf)
mt.val.mod <- lmer(y ~ mt + (-1 + mt|id), data= val.df.inf)




aics.val.infs <- c(AIC(euclidean.val.mod),
                   AIC(temporal.val.mod),
                   AIC(comp.val.mod),
                   AIC(mt.val.mod))
val.aic.diffs <- c(AIC(euclidean.val.mod) - min(aics.val.infs),
                   AIC(temporal.val.mod) - min(aics.val.infs),
                   AIC(comp.val.mod) - min(aics.val.infs),
                   AIC(mt.val.mod) - min(aics.val.infs))



val.names <- c("Spatial", "Temporal", "Spatio-temporal", "None")
val.colors <- c("#e4a021", "#4ab4da", "#2f9893", "#5d6264")
val.comparison.df <- data.frame(colors = val.colors, names= val.names, aic= val.aic.diffs)


p<-ggplot(data=val.comparison.df, aes(x=names, y=aic)) +
  geom_bar(stat="identity", fill=val.colors, width=0.65) +
  theme_minimal() +
  xlab("") +
  ylab(TeX("$\\Delta$ AIC")) +
  scale_x_discrete(limits = val.names) +
  theme(aspect.ratio = 1.2, axis.text.x = element_text(size=20, angle=40, vjust = 1.18, hjust=1.1, colour = val.colors), 
        axis.text.y=element_text(size=20), axis.title=element_text(size=25)) #+
p



################
#################
## Now we perform LOO cv to prepare for the model frequency analysis


######## LOO CV for model frequency plot

LOO.CV.reg <- function(dat, preds){
  y <- dat$y
  N <- length(y)
  subj <- dat$subj
  all.idx <- seq(N)
  loss <- matrix(nrow=N, ncol=length(colnames(preds)))
  progress <- 1
  steps <- N*length(colnames(preds))
  for (i in seq(1:N)){
    train.idx <- all.idx[-i]
    test.idx <- all.idx[i]
    
    for (j in 1:length(colnames(preds))){
      name <- colnames(preds)[j]
      train.data <- data.frame(y=y[train.idx], X = preds[train.idx, name], id = subj[train.idx])
      test.data <- data.frame(y=y[test.idx], X = preds[test.idx, name], id = subj[test.idx])
      mod <- lmer(y~ X + (-1 + X|id), data=train.data)
      
      loo.prediction <- predict(mod, newdata=test.data, type="response")
      outcome <- y[test.idx]
      log.likelihood <- dnorm(x = outcome, mean = loo.prediction, sd = sigma(mod), log = TRUE)
      
      #mse <- ((loo.prediction - outcome)**2)
      loss[i, j] <- log.likelihood
      
      cat("\r",progress/steps)
      progress <- progress + 1
      
    }
  }
  loss <- as.data.frame(loss)
  colnames(loss) <- colnames(preds)
  loss
}





#####################################################
## again we start with the non-inference objects



model.analysis.val <- data.frame(y=scale(y.val.noninf), subj = val.id.noninf)

final.predictors.val <- data.frame(euc = scale(euc.val.pred.noninf),
                                   temp = scale(temporal.val.pred.noninf), comp = scale(comp.val.pred.noninf), mt = scale(mt.val.pred.noninf))

losses.val <- LOO.CV.reg(model.analysis.val, final.predictors.val)

val.losses.df <- data.frame(losses.val, id = val.id.noninf)
write.csv(val.losses.df, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\loocv_value_models_noninf.csv", row.names=F)

###################################################
### now we turn to the inference objects


model.analysis.val <- data.frame(y=scale(y.val.inf), subj = val.id.inf)
final.predictors.val <- data.frame( euc = scale(euc.val.pred.inf),
                                   temp = scale(temporal.val.pred.inf), comp = scale(comp.val.pred.inf), mt = scale(mt.val.pred.inf))

losses.val <- LOO.CV.reg(model.analysis.val, final.predictors.val)



val.losses.df <- data.frame(losses.val)
val.losses.df <- data.frame(losses.val, id=val.id.inf)
write.csv(val.losses.df, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\loocv_value_models_inf.csv", row.names=F)


#########################################################
### Here we plot the model frequency analysis nicely
### We start with the non-inference objects
val.freq.noninf <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\val_models_freq_noninf.csv"))


pxp.colors <- c("#e4a021", "#4ab4da", "#2f9893", "#5d6264")
models <- c('Spatial', 'Temporal', 'Spatio-temporal', 'None')
mean <- val.freq.noninf$mean
se <- sqrt(val.freq.noninf$var)
best.idx <- which.max(val.freq.noninf$xp)
exc <- rep(0, 4)
exc[best.idx] <- val.freq.noninf$xp[best.idx]
#exc.line <- c(exc[best.idx], exc[best.idx], exc[best.idx], exc[best.idx])
exc.line <- c(1, 1, 1, 1)
exc.pointer <- c(best.idx, best.idx, best.idx, best.idx)
exc.pointer.y <- c(1.001, 0.95, 0.9, 0.9)

pxp.df <- data.frame(models, mean, se, exc, exc.line, exc.pointer, exc.pointer.y)
limits <- aes(ymax = mean + se, ymin=mean - se)
exc[best.idx]


### best plot

ggplot(pxp.df, aes(x = models, y=mean)) +
  geom_bar(stat="identity", fill=pxp.colors, width=0.65) +
  geom_errorbar(limits, position="dodge", width=0.2, size=1.3) +
  geom_line(aes(x=models, y = exc.line, group = 1), linetype="solid",size=1.2, colour = "#2b3852") +
  geom_line(aes(x=exc.pointer, y = exc.pointer.y, group = 1), linetype="solid",size=1.2, colour = "#2b3852") +
  #coord_cartesian(ylim=c(0, 1.2)) +
  
  # #geom_text(aes(x=3.3, y=0.93, label=("$P_{exc}=0.995$")), size=5) +
  # #  geom_abline(intercept = comp.exc, slope = 0, linetype="dashed", size=2.5) +
  theme_bw()+
  theme(aspect.ratio = 1.4, panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border=element_blank(), 
        axis.text.x=element_text(size=20, angle=40, vjust = 1.08, hjust=1.08, colour = pxp.colors),
        axis.text.y=element_text(size=18),
        axis.title = element_text(size=18),
        plot.title = element_text(size = 18, face = "bold"))+
  
  scale_x_discrete(limits = models) +
  scale_y_continuous(expand = c(0, 0), limit=c(0.,  1.2), breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
  xlab("") + 
  ylab("Model frequency") +
  annotate("text", x=2.5, y=1.1, label=TeX("$P_{exc} = 0.995", output='character'), parse=TRUE, size=6) +
  annotate(x=0, xend=0, y=0, yend=1., colour="black", lwd=1.75, geom="segment")+
  annotate(x=0, xend=4.5, y=0, yend=0, colour="black", lwd=1., geom="segment")


###

######################
## now we turn to the inference objects


val.freq.inf <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\val_models_freq_inf.csv"))
individual.p.inf <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\individual_pxp_inf.csv"))


pxp.colors <- c("#e4a021", "#4ab4da", "#2f9893" ,  "#5d6264")
models <- c('Spatial', 'Temporal', 'Spatio-temporal', "None")
mean <- val.freq.inf$mean
se <- sqrt(val.freq.inf$var)
best.idx <- which.max(val.freq.inf$xp)
exc <- rep(0, 4)
exc[best.idx] <- val.freq.inf$xp[best.idx]
exc.line <- c(exc[best.idx], exc[best.idx], exc[best.idx])
exc.line <- c(1, 1, 1, 1)
exc.pointer <- c(best.idx, best.idx, best.idx, best.idx)
exc.pointer.y <- c(1.001, 0.95, 0.9, 0.9)
limits <- aes(ymax = mean + se, ymin=mean - se)
pxp.df <- data.frame(models, mean, se, exc, exc.line, exc.pointer, exc.pointer.y)

exc[best.idx]


## best

ggplot(pxp.df, aes(x = models, y=mean)) +
  geom_bar(stat="identity", fill=pxp.colors, width=0.65) +
  geom_errorbar(limits, position="dodge", width=0.2, size=1.3) +
  geom_line(aes(x=models, y = exc.line, group = 1), linetype="solid",size=1.2, colour = "#2b3852") +
  geom_line(aes(x=exc.pointer, y = exc.pointer.y, group = 1), linetype="solid",size=1.2, colour = "#2b3852") +
  #coord_cartesian(ylim=c(0, 1.2)) +
  
  # #geom_text(aes(x=3.3, y=0.93, label=("$P_{exc}=0.995$")), size=5) +
  # #  geom_abline(intercept = comp.exc, slope = 0, linetype="dashed", size=2.5) +
  theme_bw()+
  theme(aspect.ratio = 1.4, panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.border=element_blank(), 
        axis.text.x=element_text(size=20, angle=40, vjust = 1.08, hjust=1.08, colour = pxp.colors),
        axis.text.y=element_text(size=18),
        axis.title = element_text(size=18),
        plot.title = element_text(size = 18, face = "bold"))+
  
  scale_x_discrete(limits = models) +
  scale_y_continuous(expand = c(0, 0), limit=c(0.,  1.2), breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1)) +
  xlab("") + 
  ylab("Model frequency") +
  annotate("text", x=2.5, y=1.1, label=TeX("$P_{exc} = 0.988", output='character'), parse=TRUE, size=6) +
  annotate(x=0, xend=0, y=0, yend=1., colour="black", lwd=1.75, geom="segment")+
  annotate(x=0, xend=4.5, y=0, yend=0, colour="black", lwd=1., geom="segment")

##





##################################
##################################
### That's the end of the computational modelling and data analysis :)
### -TS
##################################
