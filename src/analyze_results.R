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

results_path <- "../results/06-03/"

# All
## file_names <- list.files(results_path,
##                         pattern="model[0-9]_n[0-9]+_k[0-9]+_m2+_results_2021-0[5-6]-([0-9]+).RData")

file_names <- list.files(results_path,
                         pattern="model[1-5]_n[0-9]+_k[0-9]+_m[0-9]+_results_2021-06-(0[7-9]+).RData")

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

  model_params <- stringr::str_match(fn, "model([1-5])_n([0-9]+)_k([0-9]+)_m([0-9])")

  model_index <- model_params[2]
  n <- model_params[3]
  k <- model_params[4]
  m <- model_params[5]
  if(is.null(samples)) {
    print(sprintf("No samples for %s", fn))
  }
  else if(!is.null(samples)) {

    beta_rmse <- stan_results %>%
      spread_draws(beta[K]) %>% group_by(.draw) %>%
      summarize(RMSE = sqrt(sum((beta - data_list$tau)^2)),
                RMSE_no_confounding = sqrt(sum((beta - data_list$obs_tau)^2)),
                RMSE_rev_confounding = sqrt(sum((beta - data_list$obs_tau - data_list$bias)^2)))

    model_df <- stan_results %>%
      spread_draws(B[K, M], gamma[M], beta[K], bias[K], alpha, r2, sigma_total) %>%
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
    model_df$sigma_total_true <- data_list$sigma_total
    model_df$r  <- as.numeric(data_list$sigma_total / sqrt(data_list$Sigma_u_t[1, 1]))


  } else {
    model_df <- data.frame()
  }
  model_df
}
)

## Gamma points / RMSE

res <- res %>% mutate(n=as.integer(n), ntreat=as.integer(ntreat))

all_gamma_plots <- plot_gammas(res,
                               filter=function(...) filter(..., m==2, model!="Stanarm defaults"),
                               facet_cols=8, facet_vars=c("ntreat", "n", "model"))

all_r2_plots <- plot_r2_histograms(res,
                                   filter=function(...) filter(..., m==2, model!="Stanarm defaults"),
                                   facet_cols=8, facet_vars=c("ntreat", "n", "model"))

ggsave(all_gamma_plots, file="../figs/all_gamma.pdf")
ggsave(all_r2_plots, file="../figs/all_r2.pdf")


gamma_plot1  <- plot_gammas(res, function(...) filter(..., m==2, ntreat==5, n==1000, model != "Stanarm defaults"), facet_cols=4, facet_vars=c("model"))
r2_plot1 <- plot_r2_histograms(res, function(...) filter(..., m==2, ntreat==5, n==1000, model != "Stanarm defaults"), facet_cols=4, facet_vars=c("model"))

combined_plots  <- gamma_plot1 / r2_plot1
ggsave(combined_plots, file="../figs/combined_latest.pdf", width=10, height=5)


gamma_plot_vert  <- plot_gammas(res,
                                function(...) filter(..., m==2, ntreat==5, n==1000, model != "Stanarm defaults"),
                                facet_cols=1, facet_vars=c("model"))
r2_plot_vert <- plot_r2_histograms(res, function(...) filter(..., m==2, ntreat==5, n==1000, model != "Stanarm defaults"), facet_cols=1, facet_vars=c("model"))
combined_plots_vert  <- gamma_plot_vert + r2_plot_vert
ggsave(combined_plots_vert, file="../figs/combined_vert.pdf", height=7, width=7)

all_r2_plots_m5 <- plot_r2_histograms(res,
                                   filter=function(...) filter(..., m==5, model!="Stanarm defaults"),
                                   facet_cols=8, facet_vars=c("ntreat", "n", "model"))
ggsave(all_r2_plots_m5, file="../figs/r2_m5.pdf")









gamma_plot_vert_nc  <- plot_gammas(res,
                                function(...) filter(..., m==2, ntreat==5, n==1000, model != "Stanarm defaults"),
                                facet_cols=1, facet_vars=c("model"), color_var=sym("RMSE_no_confounding")) + ggtitle("No confounding") + labs(color="RMSE")
gamma_plot_vert_rev  <- plot_gammas(res,
                                function(...) filter(..., m==2, ntreat==5, n==1000, model != "Stanarm defaults"),
                                facet_cols=1, facet_vars=c("model"), color_var=sym("RMSE_rev_confounding")) + ggtitle("Opposite Bias") + labs(color="RMSE")

dgp_plot <- (gamma_plot_vert + ggtitle("No Treatment Effects") + theme(legend.position="none")) + (gamma_plot_vert_nc + theme(legend.position="none")) + gamma_plot_vert_rev
ggsave(dgp_plot, file="../figs/dgp_plot.pdf")



## M = 5

plot_gammas(res,
            filter=function(...) filter(..., m==5, ntreat < 20, model!="Stanarm defaults", model!="Gamma param"),
            facet_cols=4, facet_vars=c("ntreat", "n", "model"))

plot_r2_histograms(res,
                   filter=function(...) filter(..., m==5, model!="Stanarm defaults", !(model=="Gamma param" & n==100 & ntreat==5)),
                   facet_cols=4, facet_vars=c("ntreat", "n", "model"))

res %>% filter(n==1000, ntreat==5, model=="Gamma param") %>% ggplot() + geom_histogram(aes(x=r2))



stan_results %>% spread_draws(bias[K]) %>% group_by(K) %>% summarize(mean(bias))

## Sigma plot
as_tibble(res) %>% filter(m==2) %>% select(sigma_total, sigma_total_true, r2, RMSE, n, ntreat, model) %>% distinct() %>%
  arrange(desc(RMSE)) %>%
  ggplot() + geom_point(aes(x=sigma_total, y=r2, col=RMSE), alpha=0.75) + theme_bw() +
  geom_vline(aes(xintercept=sigma_total_true), linetype="dashed") +
  facet_wrap(~ n + ntreat + model, ncol=5, labeller = label_wrap_gen(multi_line=FALSE)) +
  scale_color_continuous_sequential(palette = "Viridis")


## Gamma points / RMSE
r2_histograms <- as_tibble(res) %>% filter(m==5) %>% select(M, gamma, r2, RMSE, n, ntreat, model) %>% distinct() %>%
arrange(desc(RMSE)) %>%
  pivot_wider(names_from=M, values_from=gamma, names_prefix="g") %>%
  ggplot() + geom_point(aes(x=g3, y=g4, col=RMSE), alpha=0.75) + theme_bw() +
  facet_wrap(~ model + n + ntreat) +
  scale_color_continuous_sequential(palette = "Viridis")
ggsave(r2_histograms, file="r2_histograms.pdf")

## R^2 Histogram
as_tibble(res) %>% filter(m==2) %>% select(M, m, gamma, r2, RMSE, n, ntreat, model) %>% distinct() %>%
arrange(desc(RMSE)) %>%
  pivot_wider(names_from=M, values_from=gamma, names_prefix="g") %>%
  ggplot() + geom_histogram(aes(x=r2, col=RMSE), alpha=0.75) + theme_bw() +
  facet_wrap(~ n + m + ntreat + model, ncol=4, labeller = label_wrap_gen(multi_line=FALSE)) +
  scale_color_continuous_sequential(palette = "Viridis")

## Beta
as_tibble(res) %>% filter(m==2) %>% select(K, m, r2, RMSE, alpha, beta, n, ntreat, model) %>% distinct() %>%
  arrange(desc(RMSE)) %>%
  mutate(ntreat = as.numeric(ntreat)) %>%
  filter(K==1 | K==ceiling(ntreat/2) + 1)
  pivot_wider(names_from=K, values_from=beta, names_prefix="beta") %>%
  ggplot() + geom_point(aes(x=beta1, y=!!paste0("beta", ceiling(data_list$k/2)),
                            col=RMSE), alpha=0.75) + theme_bw() +
  facet_wrap(~ n + ntreat + model, ncol=5, labeller = label_wrap_gen(multi_line=FALSE)) +
    scale_color_continuous_sequential(palette = "Viridis")

as_tibble(res) %>% filter(m==2) %>% select(K, m, r2, RMSE, alpha, beta, n, ntreat, model) %>% distinct() %>%
  arrange(desc(RMSE)) %>%
  mutate(ntreat = as.numeric(ntreat)) %>%
  pivot_wider(names_from=K, values_from=beta, names_prefix="beta") %>%
  ggplot() + geom_point(aes(x=beta1, y=alpha,
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

loo_list  <- lapply(file_names, function(fn) {
  load(paste0(results_path, fn))
  loo(stan_results)
})

## n=100, k=5
pat  <- "n100_k5"
loo_sub_list  <- loo_list[grep(pat, file_names)]
names(loo_sub_list)  <- file_names[grep(pat, file_names)]
loo::loo_compare(loo_sub_list)


pat  <- "n100_k10"
loo_sub_list  <- loo_list[grep(pat, file_names)]
names(loo_sub_list)  <- file_names[grep(pat, file_names)]
loo::loo_compare(loo_sub_list)


pat  <- "n1000_k5"
loo_sub_list  <- loo_list[grep(pat, file_names)]
names(loo_sub_list)  <- file_names[grep(pat, file_names)]
loo::loo_compare(loo_sub_list)


pat  <- "n1000_k"
loo_sub_list  <- loo_list[grep(pat, file_names)]
names(loo_sub_list)  <- file_names[grep(pat, file_names)]
loo::loo_compare(loo_sub_list)
