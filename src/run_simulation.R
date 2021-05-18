library(tidyverse)
library(mvtnorm)
library(rstiefel)
library(lubridate)
library(CopSens)
library(rstan)
library(posterior)
library(shinystan)
library(tidybayes)
library(ggridges)
library(colorspace)
options(mc.cores = parallel::detectCores())
library(R.utils)
source("utility.R")

argv <- R.utils::commandArgs(trailingOnly=TRUE, asValues=TRUE)

sigma_y_prior <- as.numeric(get_attr_default(argv, "sigma_y_prior", 1))
beta_prior <- as.numeric(get_attr_default(argv, "beta_prior", 1))


print(sprintf("Using mscale: %s", mscale))
print(sprintf("Using escale: %s", escale))

seed  <- 104879
data_list  <- generate_sparse_data(n=300, k=20, m=2, sparsity=0.4, seed=seed)

sm <- stan_model("stan/naive_flat_priors.stan.stan")


sm <- stan_model("stan/gamma_param.stan")

sm3 <- stan_model("stan/regression_r2_param.stan")

sm_horseshoe <- stan_model("stan/regression_horseshoe.stan")

sm_horseshoe <- stan_model("stan/regression_horseshoe.stan")

y  <- as.numeric(data_list$y)
X  <- apply(data_list$t, 2, function(x) scale(x, center=FALSE))
N  <- length(y)
K  <- ncol(X)
M  <-  data_list$m

stan_data  <-  list(N=N, K=K, M=M, X=X, y=y)
stan_results  <- sampling(sm_horseshoe, data=stan_data, chains=4,
                          control=list(adapt_delta=0.9, max_treedepth=13))

save(seed, data_list, stan_data, stan_results,
     file=sprintf("../results/regress_n%i_k%i_m%i_horshoe_results_%s.RData", N, K, M, lubridate::today()))
