library(lmerTest)
library(lme4)
library(ggplot2)
library(caret)
library(reshape2)


## open simulation datasets datasets
sr_gp_sim <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\simulations\\sr_gp_sim.csv"))
euc_sim <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\simulations\\euclidean_sim.csv"))
comp_sim <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\simulations\\compositional_sim.csv"))
MT_sim <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\simulations\\mean_tracker_sim.csv"))
sr_gp_optimized_sim <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\simulations\\sr_gp_sim_optimized.csv"))

### open validation datasets
sr_gp_val <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\validations\\SR-GP-validation.csv"))
euc_val <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\validations\\Euclidean-validation.csv"))
comp_val <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\validations\\Compositional-validation.csv"))
MT_val <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\validations\\Mean-tracker-validation.csv"))
sr_gp_optimized_val <- as.data.frame(read.csv("C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\validations\\SR-GP-optimized-validation.csv"))


sim.list <- list(sr_gp_sim, euc_sim, comp_sim, MT_sim, sr_gp_optimized_sim)
val.list <- list(sr_gp_val, euc_val, comp_val, MT_val, sr_gp_optimized_val)
colnames(sr_gp_val)

compute.validation.scores <- function(y, model_predictors, model_names){
  nll <- rep(0, length(model_names))
  c <- 1
  for (name in model_names){
    X <- scale(model_predictors[, name])
    val.model <- glm(y~ -1 + X, family="binomial")
    nll[c] <- -logLik(val.model)
    c <- c+1
  }
  
  nll
}


plot.matrix <- function(matrix, names){
  D <- melt(matrix)
  p <- ggplot(data=D,aes(x=(Var1), y=(Var2), fill=value)) + geom_tile() +
    scale_fill_gradient(low="white", high="blue")+  theme(axis.text=element_text(size=12), axis.title=element_text(size=14,face="bold")) +
    geom_text(aes(Var1, Var2, label = round(value)), color = "black", size = 4)
  
  p
  
}
compute.aic <- function(nll, p){
  ll <- -nll
  (p*2) - (2*ll)
}
compute.weights <- function(scores){
  
  min.score <- min(scores)
  diff.scores <-rep(0, length(scores))
  c <- 1
  
  for (score in scores){
    diff <- score - min.score
    diff.scores[c] <- diff
    c <- c+1
  }
  weights <-rep(0, length(scores))
  c <- 1
  for (diff in diff.scores){
    w <- exp(-0.5 * diff) / sum(exp(-0.5*diff.scores))
    weights[c] <- w
    c <- c+1
  }
  
  list("diff" = diff.scores, "weights" = weights)
  
}

compute.posterior <- function(nll_matrix){
  posterior_matrix <- matrix(0, nrow = nrow(nll_matrix), ncol = ncol(nll_matrix))
  c <- 1
  p <- 1  # set number of parameters
  for (i in 1:nrow(nll_matrix)){
    nll <- nll_matrix[i, ]
    aic <- (p*2) - (2*-nll)  # compute aic
    # compute aic differences
    diff <- rep(0, length(aic))
    for (j in 1:length(aic)){
      diff[j] <- aic[j] - min(aic)
    }
    # approximate posterior with AIC weights
    posterior <- exp(-0.5*diff) / sum(exp(-0.5*diff))
    print(diff)
    print(posterior)
    posterior_matrix[c, ] <- posterior
    c <- c+1
  }
  posterior_matrix
}


validation.matrix <- matrix(0, nrow=length(sim.list), ncol = length(val.list))

for (i in 1:length(sim.list)){
  sim.data <- sim.list[[i]]
  y <- sim.data$choice_indices
  val.data <- val.list[[i]]
  
  nll <- compute.validation.scores(y, val.data, colnames(val.data))
  validation.matrix[i, ] <- nll
  
}

validation.matrix
posterior.matrix <- compute.posterior(validation.matrix)
posterior.matrix

write.csv(validation.matrix, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\nll_matrix.csv", row.names=FALSE)
write.csv(posterior.matrix, "C:\\Users\\Tankred\\Documents\\Emacs\\Notebooks\\Papers to read\\rewards in mental maps\\model_simulations\\posterior_matrix.csv", row.names = F)


plot.matrix(validation.matrix, names=colnames(sr_gp_val))
