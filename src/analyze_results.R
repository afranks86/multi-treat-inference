library(CopSens)
library(rstan)
library(posterior)
library(shinystan)
library(tidybayes)
library(ggridges)
library(colorspace)
options(mc.cores = parallel::detectCores())

source("utility.R")
seed  <- 104879

results_path <- "../results/05-25/"
file_names <- list.files(results_path,
                        pattern="model[0-9]_n[0-9]+_k[0-9]+_m[0-9]+_results_2021-05-(2[1-5]).RData")

get_gamma <- function(df, Bref) {
  B <- df %>% pull(B) %>% matrix(ncol=data_list$m, byrow=TRUE)
  svd_rot <- svd(t(Bref) %*% B)
  R <- svd_rot$u %*% t(svd_rot$v)
  df$gamma <- as.numeric(R %*% t(matrix(df$gamma, ncol=data_list$m, byrow=TRUE)))
  df
}

res <- purrr::map_dfr(file_names, function(fn) {

  ## Print the name of the file
  print(fn)
  load(paste0(results_path, fn), .GlobalEnv)
  samples <- rstan::extract(stan_results)

  model_params <- stringr::str_match(fn, "model([0-9])_n([0-9]+)_k([0-9]+)_m([0-9])")

  model_index <- model_params[2]
  n <- model_params[3]
  k <- model_params[4]
  m <- model_params[5]

  if(!is.null(samples)) {

    beta_rmse <- stan_results %>%
      spread_draws(beta[K]) %>% group_by(.draw) %>%
      summarize(RMSE = sqrt(sum((beta - data_list$tau)^2)))

    model_df <- stan_results %>%
      spread_draws(B[K, M], gamma[M], beta[K], bias[K], r2) %>%
      group_by(.draw) %>%
      nest() %>%
      mutate(data = map(data, function(d) get_gamma(d, t(data_list$B)))) %>%
      unnest(cols = c(data)) %>% left_join(beta_rmse)

    model_df$model <- model_index
    model_df$model=recode(model_df$model,
                          `1`="Stanarm defaults",
                          `2`="Gamma param",
                          `3`="R2 param",
                          `4` = "Sparsity",
                          `5` = "Null controls")


    model_df$n <- n
    model_df$ntreat <- k
    model_df$m  <- m


  } else {
    model_df <- data.frame()
  }
  model_df
}
)

## Gamma points / RMSE
as_tibble(res) %>% filter(m==2) %>% select(M, gamma, r2, RMSE, n, ntreat, model) %>% distinct() %>%
arrange(desc(RMSE)) %>%
  pivot_wider(names_from=M, values_from=gamma, names_prefix="g") %>%
  ggplot() + geom_point(aes(x=g1, y=g2, col=RMSE), alpha=0.75) + theme_bw() +
  facet_wrap(~ n + ntreat + model, ncol=5, labeller = label_wrap_gen(multi_line=FALSE)) +
  scale_color_continuous_sequential(palette = "Viridis")


## Gamma points / RMSE
as_tibble(res) %>% filter(m==5) %>% select(M, gamma, r2, RMSE, n, ntreat, model) %>% distinct() %>%
arrange(desc(RMSE)) %>%
  pivot_wider(names_from=M, values_from=gamma, names_prefix="g") %>%
  ggplot() + geom_point(aes(x=g3, y=g4, col=RMSE), alpha=0.75) + theme_bw() +
  facet_wrap(~ model + n + ntreat) +
  scale_color_continuous_sequential(palette = "Viridis")


## R^2 Histogram
as_tibble(res) %>% filter(m==2) %>% select(M, m, gamma, r2, RMSE, n, ntreat, model) %>% distinct() %>%
arrange(desc(RMSE)) %>%
  pivot_wider(names_from=M, values_from=gamma, names_prefix="g") %>%
  ggplot() + geom_histogram(aes(x=r2, col=RMSE), alpha=0.75) + theme_bw() +
  facet_wrap(~ n + m + ntreat + model, ncol=5, labeller = label_wrap_gen(multi_line=FALSE)) +
  scale_color_continuous_sequential(palette = "Viridis")

## Beta
as_tibble(res) %>% filter(m==2) %>% select(K, m, r2, RMSE, beta, n, ntreat, model) %>% distinct() %>%
  arrange(desc(RMSE)) %>%
  mutate(ntreat = as.numeric(ntreat)) %>%
  filter(K==1 | K==ceiling(ntreat/2) + 1)
  pivot_wider(names_from=K, values_from=beta, names_prefix="beta") %>%
  ggplot() + geom_point(aes(x=beta1, y=!!paste0("beta", ceiling(data_list$k/2)),
                            col=RMSE), alpha=0.75) + theme_bw() +
  facet_wrap(~ n + ntreat + model, ncol=5, labeller = label_wrap_gen(multi_line=FALSE)) +
  scale_color_continuous_sequential(palette = "Viridis")


as_tibble(res) %>% filter(m==2) %>% select(K, m, r2, RMSE, beta, n, ntreat, model) %>% distinct() %>%
  arrange(desc(RMSE)) %>%
  mutate(ntreat = as.numeric(ntreat)) %>%
  pivot_wider(names_from=K, values_from=beta, names_prefix="beta") %>%
  ggplot() + geom_point(aes(x=beta1, y=beta2, col=RMSE), alpha=0.75) + theme_bw() +
  facet_wrap(~ n + ntreat + model, ncol=5, labeller = label_wrap_gen(multi_line=FALSE)) +
  scale_color_continuous_sequential(palette = "Viridis")


as_tibble(res) %>% filter(model==3) %>% select(K, r2, RMSE, beta, n, ntreat, model) %>% distinct() %>%
arrange(desc(RMSE)) %>%
  pivot_wider(names_from=K, values_from=beta, names_prefix="beta") %>%
  ggplot() + geom_density_2d_filled(aes(x=beta1, y=beta2), alpha=0.75) + theme_bw() +
  facet_wrap(~ model + n + ntreat) +
  scale_color_continuous_sequential(palette = "Viridis")


as_tibble(res) %>% filter(model %in% c(3, 5)) %>% select(K, r2, RMSE, beta, n, ntreat, model) %>% distinct() %>%
arrange(desc(RMSE)) %>%
  ggplot() + geom_density_ridges(stat="binline", aes(x=beta, y=model), alpha=0.75) + theme_bw() +
  facet_wrap(~ K) +
  scale_color_continuous_sequential(palette = "Viridis")
