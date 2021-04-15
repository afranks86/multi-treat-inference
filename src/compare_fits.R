library(CopSens)
library(rstan)
library(posterior)
library(shinystan)
library(tidybayes)
library(ggridges)
options(mc.cores = parallel::detectCores())

source("utility.R")
seed  <- 427851
data_list  <- generate_sparse_data(n=300, k=20, m=5, sparsity=0.4, seed=seed)

y  <- as.numeric(data_list$y)
X  <- data_list$t
N  <- length(y)
K  <- ncol(X)
M  <-  5


load("../results/regress_5_factor_results_2021-04-12.RData")
stan_results_naive <- stan_results

load("../results/regress_5_factor_results_r2_param_2021-04-12.RData")
stan_results_r2  <- stan_results

beta_draws_naive  <- stan_results_naive %>% spread_draws(beta[K], r2) %>%
  mutate(type= "Naive")
beta_draws_r2  <- stan_results_r2 %>% spread_draws(beta[K], r2) %>%
  mutate(type= "R2 Param")
beta_draws  <- bind_rows(beta_draws_naive, beta_draws_r2)

beta_draws

gamma_draws_naive  <- stan_results_naive %>% spread_draws(gamma[K], r2) %>%
  mutate(type= "Naive")
gamma_draws_r2  <- stan_results_r2 %>% spread_draws(gamma[K], r2) %>%
  mutate(type= "R2 Param")
gamma_draws  <- bind_rows(gamma_draws_naive, gamma_draws_r2)


r2_beta_summaries = stan_results_r2 %>%
  spread_draws(beta[K]) %>%
  median_qi(estimate = beta) %>%
  mutate(model = "R2 Param")

naive_beta_summaries = stan_results_naive %>%
  spread_draws(beta[K]) %>%
  median_qi(estimate = beta) %>%
  mutate(model = "Naive Param")

bind_rows(r2_beta_summaries, naive_beta_summaries) %>%
  ggplot(aes(y = K, x = estimate, xmin = .lower, xmax = .upper, color = model)) +
  geom_pointinterval(position = position_dodge(width = .3))  + theme_bw()
