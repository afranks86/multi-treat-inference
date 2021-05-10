library(CopSens)
library(rstan)
library(posterior)
library(shinystan)
library(tidybayes)
library(ggridges)
options(mc.cores = parallel::detectCores())

source("utility.R")
seed  <- 427851
data_list  <- generate_sparse_data(n=300, k=20, m=2, sparsity=0.4, seed=seed)
## data_list  <- generate_data_null(n=300, k=20, m=2, seed=seed)

plot(data_list$tau, coef(lm(data_list$y ~ data_list$t - 1)))
cor(data_list$tau, coef(lm(data_list$y ~ data_list$t - 1)))


sm <- stan_model("stan/regression2.stan")
sm2 <- stan_model("stan/regression_stanarm_defaults.stan")
sm3 <- stan_model("stan/regression_r2_param.stan")
sm_horseshoe <- stan_model("stan/regression_horseshoe.stan")

y  <- as.numeric(data_list$y)
X  <- apply(data_list$t, 2, function(x) scale(x, center=FALSE)
N  <- length(y)
K  <- ncol(X)
M  <-  data_list$m

stan_data  <-  list(N=N, K=K, M=M, X=X, y=y)
stan_results  <- sampling(sm_horseshoe, data=stan_data, chains=4,
                          adapt_delta=0.9, treedepth=13)

save(seed, stan_data, stan_results, file=sprintf("../results/regress_5_factor_results_%s.RData", lubridate::today()))


stan_results  <- sampling(sm3, data=stan_data, chains=4)

save(seed, stan_data, stan_results, file=sprintf("../results/regress_5_factor_results_r2_param_%s.RData", lubridate::today()))


stan_data2 <- stan_data
stan_data2$X = scale(X, center=FALSE)
stan_results  <- sampling(sm4, data=stan_data2, chains=4)

samples <- extract(stan_results)
min_index  <- which.min(apply(samples$beta, 1, function(x) median(abs(x))))


samples$r2[min_index]
cor.test(data_list$tau, samples$beta[min_index, ])

launch_shinystan(stan_results)

stan_results %>% spread_draws(beta[K], r2) %>%
  mutate(K = as.factor(K)) %>%
ggplot() + geom_point(aes(x=r2, y=beta, col=r2), size=0.1) +
  facet_wrap(~K, scales="free")

stan_results %>% spread_draws(beta[K], r2) %>%
  group_by(.draw) %>%
  summarize(norm = sqrt(sum(beta^2))) %>%
  ungroup %>%
  ggplot() + geom_histogram(aes(x=norm))




stan_results %>% spread_draws(beta[K], r2) %>%
  mutate(K = as.factor(K)) %>%
ggplot() + geom_histogram(aes(beta)) +
  facet_wrap(~K, scales="free")

stan_results %>% spread_draws(beta[K], r2) %>%
  ggplot() + geom_histogram(aes(x=r2), bins=50)

stan_results %>% spread_draws(beta[K]) %>%
  mutate(K=as.factor(K)) %>% median_qi(beta)

stan_results %>% spread_draws(beta[K], r2) %>%
  filter(r2 > 0.8) %>%
  mutate(K=as.factor(K)) %>% median_qi(beta)

stan_results %>% spread_draws(gamma[M], r2) %>%
  pivot_wider(names_from=M, values_from=gamma, names_prefix="g") %>%
  ggplot() + geom_point(aes(x=g1, y=g2, col=r2))

stan_results %>% spread_draws(beta[K]) %>%
  ggplot(aes(x=beta, y=as.factor(K))) + geom_density_ridges()


stan_results %>% spread_draws(beta[K]) %>% pull(K) %>% table
