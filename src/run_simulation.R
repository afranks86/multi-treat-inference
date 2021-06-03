library(tidyverse)
library(mvtnorm)
library(rstiefel)
library(lubridate)
library(rstan)
library(ggridges)
library(colorspace)
options(mc.cores = parallel::detectCores())
library(R.utils)
source("utility.R")

get_attr_default <- function(thelist, attrname, default) {
    if(!is.null(thelist[[attrname]])) thelist[[attrname]] else default
}

argv <- R.utils::commandArgs(trailingOnly=TRUE, asValues=TRUE)

model <- as.numeric(get_attr_default(argv, "model", 1))
n <- as.numeric(get_attr_default(argv, "n", 100))
k <- as.numeric(get_attr_default(argv, "k", 10))
m <- as.numeric(get_attr_default(argv, "m", 2))

seed  <- 104879
data_list  <- generate_data_null(n=n, k=k, m=m, seed=seed)

y  <- as.numeric(data_list$y)
X  <- apply(data_list$t, 2, function(x) scale(x, center=FALSE))
N  <- length(y)
K  <- ncol(X)
M  <-  data_list$m

print(sprintf("model=%i, n=%i, k=%i, m=%i", model, n, k, m))

stan_data  <-  list(N=N, K=K, M=M, X=X, y=y)

if(model == 1){
    print("Running stanarm defaults model.")
    sm <- stan_model("stan/stanarm_defaults.stan")
    stan_data$alpha_scaler = 1
    stan_data$beta_scaler = 1
} else if (model == 2) {
    print("Running gamma param model.")
    sm <- stan_model("stan/gamma_param.stan")
} else if(model == 3) {
    print("Running r2 param model.")
    sm <- stan_model("stan/r2_param.stan")
} else if (model == 4) {
    print("Running horseshoe model")
    sm <- stan_model("stan/horseshoe.stan")
    stan_data$frac_non_null <- 0.2
    stan_data$slab_scale <- 3
} else if (model == 5) {
  sm <- stan_model("stan/null_controls.stan")
  stan_data$num_null  <- 1
  stan_data$null_control_indices <- as.array(c(1))
  stan_data$non_null_control_indices <- setdiff(1:K, stan_data$null_control_indices)
}

stan_results  <- sampling(sm, data=stan_data, chains=4,
                          control=list(adapt_delta=0.9, max_treedepth=13))

save(seed, data_list, stan_data, stan_results,
     file=sprintf("../results/model%i_n%i_k%i_m%i_results_%s.RData", model, N, K, M, lubridate::today()))
