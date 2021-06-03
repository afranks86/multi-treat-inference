library(CopSens)
library(rstan)
library(posterior)
library(shinystan)
library(tidybayes)
library(ggridges)
options(mc.cores = parallel::detectCores())

mice_data <- read_csv("../data/micedata.csv")

sm_horseshoe <- stan_model("stan/horseshoe.stan")

y  <- mice_data %>% pull(y)
X  <- mice_data %>% select(Igfbp2:Veph1) %>% mutate_all(function(x) scale(x, center=FALSE))

N  <- length(y)
K  <- ncol(X)
M  <-  3

stan_data  <-  list(N=N, K=K, M=M, X=as.matrix(X), y=y)
stan_data$frac_nonzero <- 0.1
stan_data$slab_scale <- 1
stan_results  <- sampling(sm_horseshoe, data=stan_data, chains=4, control=list(adapt_delta=0.9, max_treedepth=13)))

save(stan_results, file="../results/mice_horseshoe.RData")

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
