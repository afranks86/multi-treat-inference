library(tidyverse)
library(rstan)
library(posterior)
library(shinystan)
library(tidybayes)
library(ggridges)
options(mc.cores = parallel::detectCores())

mice_data <- read_csv("../data/micedata.csv")

#sm_gp <- stan_model("stan/gp_model_no_confounding.stan")
sm_gp <- stan_model("stan/gp_model.stan")

y  <- mice_data %>% pull(y)
X  <- mice_data %>% select(Igfbp2:Veph1) %>% mutate_all(function(x) scale(x, center=FALSE))

N  <- length(y)
K  <- ncol(X)
M  <-  3

## outcome model
## Evaluate at quantiles of the treatment
percentiles  <- c(0.025, 0.975)
Xpred <- matrix(0, nrow=0, ncol=ncol(X))
gene_mat <- matrix(apply(X, 2, median), nrow=length(percentiles), ncol=ncol(X), byrow=TRUE)
for(i in 1:ncol(X)) {
  gene_mat_i <- gene_mat
  gene_mat_i[, i]  <- as.numeric(quantile(X[i,], probs=percentiles))
  Xpred <- rbind(Xpred, gene_mat_i)
}
colnames(Xpred)  <-  colnames(X)
rownames(Xpred) <- rep(colnames(X), each=length(percentiles))


stan_data  <-  list(N=N, Npred=nrow(Xpred), M=M,
                    X=as.matrix(X), Xpred=Xpred,
                    y=y, K=K)
stan_results  <- rstan::sampling(sm_gp, data=stan_data, chains=2)

saveRDS(stan_results, file="gp_results.RDS")

