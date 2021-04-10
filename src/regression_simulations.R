library(CopSens)
library(rstan)
library(posterior)
library(shinystan)
library(tidybayes)
library(ggridges)
options(mc.cores = parallel::detectCores())

source("utility.R")
seed  <- 757221
data_list  <- generate_sparse_data(n=300, k=20, m=1, sparsity=0.4, seed=seed)

plot(data_list$tau, coef(lm(data_list$y ~ data_list$t - 1)))
cor(data_list$tau, coef(lm(data_list$y ~ data_list$t - 1)))


sm <- stan_model("stan/regression2.stan")

y  <- as.numeric(data_list$y)
X  <- data_list$t
N  <- length(y)
K  <- ncol(X)
M  <-  1

stan_data  <-  list(N=N, K=K, M=M, X=X, y=y)
stan_results  <- sampling(sm, data=stan_data, chains=4)

save(seed, stan_data, stan_results, file=sprintf("../results/regress_results_%s.RData", lubridate::today()))

launch_shinystan(stan_results)

stan_results %>% spread_draws(beta[K], r2) %>%
  mutate(K = as.factor(K)) %>%
ggplot() + geom_point(aes(x=r2, y=beta, col=r2), size=0.1) +
  facet_wrap(~K, scales="free")


stan_results %>% spread_draws(beta[K], r2) %>%
  mutate(K = as.factor(K)) %>%
ggplot() + geom_histogram(aes(beta)) +
  facet_wrap(~K, scales="free")

stan_results %>% spread_draws(beta[K], r2) %>%
  ggplot() + geom_histogram(aes(x=r2))

stan_results %>% spread_draws(beta[K]) %>%
  mutate(K=as.factor(K)) %>% median_qi(beta)

stan_results %>% spread_draws(beta[K], r2) %>%
  filter(r2 > 0.8) %>%
  mutate(K=as.factor(K)) %>% median_qi(beta)


stan_results %>% spread_draws(beta[K]) %>%
  ggplot(aes(x=beta, y=as.factor(K))) + geom_density_ridges()


stan_results %>% spread_draws(beta[K]) %>% pull(K) %>% table
