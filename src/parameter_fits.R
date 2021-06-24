library(lme4)
library(lmerTest)
library(ggplot2)
library(car)
library(caret)


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
temporal <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\sr_gp_weighted_results3.csv"))

compositional <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\comp_results_v3.csv"))

################
## optional models, only semi-relevant:
optimized_gp <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\optimized_results.csv"), header=FALSE)
optimized_gp <- optimized_gp[,-1]  ## remove index column

## SR Softmax
sr_softmax <-as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\softmax_sr.csv"), header=FALSE)
 
## mean tracker and random sr kernel
mean_tracker <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\mean_tracker.csv"), header=FALSE)
comp_rand_sr <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\comp_random_SR.csv"), header=FALSE)
srgp_rand <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\SR_GP_random.csv"), header=FALSE)
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
## optimized gp
##

op_gp_mod <- glmer(y~ -1 + optimized_gp + (-1 + optimized_gp|subj), family="binomial")
logLik(op_gp_mod)




##### Mean tracker
MT <- scale(mean_tracker)
MT_mod <- glmer(y~-1 + MT +(-1 + MT|subj), family="binomial")
MT_nll <- -logLik(MT_mod)



#### compositional with random SR-kernel

comp_rand <- scale(comp_rand_sr)
comp_rand_mod <- glmer(y~-1 + comp_rand + (-1 + comp_rand|subj), family="binomial")
logLik(comp_rand_mod)
summary(comp_rand_mod)

### compositional with half identity matrix
comp_id <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\comp_identity.csv"), header=FALSE)

comp_id <- scale(comp_id)
comp_rand_mod <- glmer(y~-1 + comp_id + (-1 + comp_id|subj), family="binomial")
logLik(comp_rand_mod)


## sr gp paired with random temporal kernels
srgp_rand <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\param_fits\\SR_GP_random.csv"), header=FALSE)

sr_rand <- scale(srgp_rand)
sr_rand_mod <- glmer(y~-1 + sr_rand + (-1 + sr_rand|subj), family="binomial")
logLik(sr_rand_mod)


###
# Compositional kernel
##


comp.output <- find.minimum(compositional, y)
comp.output$min.nll
comp.output$best.param
comp.output$idx



param.grid.comp <- expand.grid(seq(0.01, 0.7, length.out = 25), seq(0.1, 4, length.out = 25))
param.grid.comp[comp.output$idx,]


comp.p <- 4  # number of free parameters

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
#write.csv(model.diagnostics.df, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\effects_and_weights.csv")


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
#write.csv(error.matrix, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\tBt_euc_w.csv", row.names=F)


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

compute.logistic.fit <- function(matrix, excluded.trials){
  trials.X <- seq(1:100)
  trials.X <- trials.X[!(trials.X %in% excluded.trials)]
  inflection.points <- rep(0, nrow(matrix))
  coefs <- rep(0, nrow(matrix))
  
  predictions <- matrix(nrow=nrow(matrix), ncol = length(trials.X))
  for (i in 1:nrow(matrix)){
    delta_w <- matrix[i, trials.X]
    fit <- glm(delta_w ~ trials.X, family=binomial(link=logit))
    k <- coef(fit)[2]
    preds <- predict(fit, type="response")
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

ggplot(inflection.df, aes(x=inflection, y=R)) + geom_point() + geom_smooth(method="lm") + theme_bw() +
  xlab("Estimated inflection point (trials)") + ylab("Mean reward") +theme(axis.text=element_text(size=12),
                                                                             axis.title=element_text(size=14,face="bold"))


cor.test(logistic.fits$inflection.point[pos.idx], m.rewards[pos.idx])

### get the top/bottom 15 subjects and look at differences in inflection points
pos.idx

top.ten <- sorted.idx[31:45]
bottom.ten <- sorted.idx[1:15]

logistic.fits.HP <- compute.logistic.fit(error.matrix[pos.idx, ][which(m.rewards[pos.idx] > mean(m.rewards)), ], excluded.trials)
logistic.fits.LP <- compute.logistic.fit(error.matrix[pos.idx, ][which(m.rewards[pos.idx] <= mean(m.rewards)), ], excluded.trials)

logistic.fits.HP <- compute.logistic.fit(error.matrix[pos.idx, ][top.ten, ], excluded.trials)
logistic.fits.LP <- compute.logistic.fit(error.matrix[pos.idx, ][bottom.ten, ], excluded.trials)
hist(logistic.fits.HP$inflection.point)
hist(logistic.fits.LP$inflection.point)

length(logistic.fits.HP$inflection.point)
t.test(logistic.fits.HP$inflection.point, logistic.fits.LP$inflection.point, equal.variances=F)
m.hp.inf <- mean(logistic.fits.HP$inflection.point)
sd.hp.inf <- sd(logistic.fits.HP$inflection.point)
m.lp.inf <- mean(logistic.fits.LP$inflection.point)
sd.lp.inf <- sd(logistic.fits.LP$inflection.point)

m <- c(m.hp.inf, m.lp.inf)
s <- c(sd.hp.inf, sd.lp.inf)

barplot(m, ylab = "Mean inflection point (trial)", ylim = c(0, 100), names.arg=c("High performing", "Low performing"))

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

random.loss <- log(0.5) * 4700

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
colors <- c("#e4a021", "#4ab4da", "#2f7c97", "#5d6264")
comparison.df <- data.frame(aic.stats$weights, aic.stats$diff, bic.stats$weights, bic.stats$diff,  names, r2.stats)


### full comparison
p<-ggplot(data=comparison.df, aes(x=names, y=aic.stats$weights)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_minimal() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, face="bold")) + ggtitle("AIC weights") +
  xlab("Models") + ylab("Weight") + ylim(0, 1) + theme(axis.text=element_text(size=12), axis.title=element_text(size=14,face="bold"), plot.title = element_text(size = 18, face = "bold"))

p


p<-ggplot(data=comparison.df, aes(x=names, y=aic.stats$diff)) +
  geom_bar(stat="identity", fill=colors, width=0.5, position = position_dodge(width = 3))+
  theme_minimal() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, face="bold")) + ggtitle("Cognitive map") + scale_x_discrete(limits = names) +
  xlab("") + ylab("AIC difference") +theme(axis.text.x=element_text(size=15, angle=40, vjust = 1, hjust=0.8), axis.text.y=element_text(size=15), axis.title=element_text(size=14,face="bold"), plot.title = element_text(size = 18, face = "bold"))
p


# this one isnt ready yet
p<-ggplot(data=comparison.df, aes(x=names, y=r2.stats)) +
  geom_bar(stat="identity", fill="steelblue")+
  theme_minimal() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + ggtitle("Pseudo R-squared") +
  xlab("Models") + ylab("R2")+ ylim(0,.5)
p


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
ideal.mod.obs <- glm(y[observed.idx] ~ -1 + true.diff[observed.idx] , family="binomial")
logLik(ideal.mod.obs)

rand.ll <- log(0.5)*length(observed.idx)
ideal.r2 <- 1 - (logLik(ideal.mod.obs) / rand.ll)
ideal.as.line <- c(ideal.r2,ideal.r2, ideal.r2, ideal.r2)
line.df <- data.frame(ideal.as.line)

comparison.df <- data.frame(aic.stats$weights, aic.stats$diff, bic.stats$weights, bic.stats$diff,  names, r2.stats, ideal.as.line)
p<-ggplot(data=comparison.df, aes(x=names, y=r2.stats)) +
  geom_bar(stat="identity", fill=colors)+
  theme_minimal() + theme(axis.text.x=element_text(size=15, angle=40, vjust = 1, hjust=0.8), axis.text.y=element_text(size=15)) + ggtitle("Pseudo R-squared") +
  xlab("Models") + ylab("R2")+ ylim(0,.5) + scale_x_discrete(limits = names) + geom_line(aes(x=ideal.as.line))

p

#######################

### check how well the other models do on this data set
### get model predictors

comp.obs <- compositional[observed.idx, comp.output$idx]  # the compositional predictor with best fitting parameters, at observed indices
euc.obs <- euc[observed.idx, euc.output$idx]  # the same for euc
sr.obs <- temporal[observed.idx, successor.output$idx]  # the same for temporal
mt.obs <- mean_tracker[observed.idx,]  # for the mean tracker

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




##################################
##################################
### That's the end of it for now:)
### (I may add stuff in the future however...)
### -TS
##################################