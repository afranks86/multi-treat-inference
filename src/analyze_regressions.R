library(CopSens)
library(rstan)
library(posterior)
library(shinystan)
library(tidybayes)
library(ggridges)
options(mc.cores = parallel::detectCores())

sm <- stan_model("stan/regression2.stan")

y  <- micedata[, 1]
X  <- micedata[, 2:ncol(micedata)]
X  <- micedata[, 2:18]

N  <- length(y)
K  <- ncol(X)
M  <-  1

stan_data  <-  list(N=N, K=K, M=M, X=X, y=y)
stan_results  <- sampling(sm, data=stan_data, chains=4)

save(stan_results, file="../results/regress_results.RData")

launch_shinystan(stan_results)

samples  <- rstan::extract(stan_results)

beta_mle  <- coef(lm(y ~ as.matrix(X)))[-1]
beta_pm  <- colMeans(samples$beta)

cor(beta_mle, beta_pm, method="spearman")

plot(beta_mle, beta_pm)

shinystan::notes

plot(samples$beta[, 1])

hist(samples$r2, breaks=50)

B <- matrix(rnorm(10*3), ncol=3)

stan_results %>% spread_draws(beta[K], r2) %>%
  filter(r2 < 0.2) %>% median_qi(beta[K])


stan_results %>% spread_draws(beta[K]) %>%
  mutate(K=as.factor(K)) %>% median_qi(beta)


stan_results %>% spread_draws(beta[K], r2) %>%
  filter(r2 > 0.1) %>%
  mutate(K=as.factor(K)) %>% median_qi(beta)



stan_results %>% spread_draws(beta[K]) %>%
  ggplot(aes(x=beta, y=as.factor(K))) + geom_density_ridges()


stan_results %>% spread_draws(beta[K]) %>% pull(K) %>% table
