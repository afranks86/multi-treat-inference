library(tidyverse)
library(mvtnorm)
library(rstiefel)
library(lubridate)
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

model <- as.numeric(get_attr_default(argv, "m", 1))
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

stan_data  <-  list(N=N, K=K, M=M, X=X, y=y)

if(model == 1){
    print("Running stanarm defaults model.")
    sm <- stan_model("stan/stanarm_defaults.stan")
} else if (model == 2) {
    print("Running gamma param model.")
    sm <- stan_model("stan/gamma_param.stan")
} else if(model == 3) {
    print("Running r2 param model.")
    sm <- stan_model("stan/r2_param.stan")
} else if (model == 4) {
    print("Running horseshoe model")
    sm <- stan_model("stan/horseshoe.stan")
} else if (model == 5) {
    sm <- stan_model("stan/null_controls.stan")
}



stan_results  <- sampling(sm,
                          data=stan_data, chains=4,
                          control=list(adapt_delta=0.9, max_treedepth=13))

save(seed, data_list, stan_data, stan_results,
     file=sprintf("../results/model%i_n%i_k%i_m%i__results_%s.RData", model, N, K, M, lubridate::today()))
