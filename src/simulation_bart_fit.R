library(tidyverse)
library(mvtnorm)
library(rstiefel)
library(lubridate)
library(rstan)
library(ggridges)
library(colorspace)
options(mc.cores = parallel::detectCores())
library(BART)
library(R.utils)
library(rstan)
library(tidybayes)
source("utility.R")

get_attr_default <- function(thelist, attrname, default) {
    if(!is.null(thelist[[attrname]])) thelist[[attrname]] else default
}

argv <- R.utils::commandArgs(trailingOnly=TRUE, asValues=TRUE)

model <- as.numeric(get_attr_default(argv, "model", 1))
n <- as.numeric(get_attr_default(argv, "n", 10000))
k <- as.numeric(get_attr_default(argv, "k", 10))
m <- as.numeric(get_attr_default(argv, "m", 2))

seed  <- 104879
data_list  <- generate_data_null(n=n, k=k, m=m, seed=seed)

Ytrain  <- as.numeric(data_list$y)
Xtrain  <- apply(data_list$t, 2, function(x) scale(x, center=FALSE))
N  <- length(data_list$y)
K  <- ncol(Xtrain)
M  <-  data_list$m


## Bayes Factor model
sm <- stan_model("stan/factor_model_homo.stan")
data_list_factor <- list()
data_list_factor$N <- N
data_list_factor$K <- K
data_list_factor$M <- 2
data_list_factor$X <- Xtrain

factor_results <- readRDS("../results/simulation_factor_results.RDS")
## factor_results <- sampling(sm, data=data_list_factor)
factor_samples <- factor_results %>% spread_draws(B[K, M], lambda[K])

n_factor_samples <- max(factor_samples$.draw)

## saveRDS(factor_results, file="../results/simulation_factor_results.RDS")

## outcome model
## Evaluate at quantiles of the treatment
test_points  <- seq(-2, 2, by=0.25)
Xtest <- matrix(0, nrow=0, ncol=ncol(Xtrain))
for(i in 1:ncol(Xtrain)) {
  pred_mat <- matrix(0, nrow=length(test_points), ncol=ncol(Xtrain), byrow=TRUE)

  pred_mat[, i]  <- test_points
  Xtest <- rbind(Xtest, pred_mat)
}
colnames(Xtest)  <-  colnames(Xtrain)
rownames(Xtest) <- rep(colnames(Xtrain), each=length(test_points))

bart_result <- gbart(y.train=Ytrain, x.train=Xtrain, x.test=Xtest, ndpost=n_factor_samples)

samples_tibble <- as_tibble(expand.grid(.draw=1:n_factor_samples, x=test_points, gene=1:K))
samples_tibble$yhat <- as.numeric(bart_result$yhat.test)
samples_tibble$sigma_total <- as.numeric(rep(bart_result$sigma[101:(n_factor_samples + 100)], ncol(bart_result$yhat.test)))

samples_tibble %>% filter(gene == 1) %>%
  group_by(x) %>%
  summarize(lower = quantile(yhat, 0.025), upper=quantile(yhat, 0.975)) %>%
  ggplot() + geom_ribbon(aes(x=x, ymin=lower, ymax=upper), alpha=0.5) + theme_bw()


## Join with factor model samples
samples_joined_one  <-  full_join(samples_tibble %>% filter(gene==1),
                              factor_samples, by=c(".draw"))

## treat_median <- median(Xtrain[, 1])
## treat_quantiles <- quantile(Xtrain[, 1], percentiles)

### Bias
for(gene in c(1, 2, 6)) {
  gene_tibble <- bias_tibble <- samples_joined %>%
    filter(gene == !!gene)

  bias_tibble <- gene_tibble %>%
    group_by(`.draw`) %>%
    nest() %>%
    transmute(bias = map(data, function(df) {
      B <- df %>% filter(x == -2) %>% pull(B) %>% matrix(., nrow=K, byrow=TRUE)
      lambda <- df %>%  filter(x == -2, M==1) %>% pull(lambda) %>% diag
      sigma_total  <- df %>% pull(sigma_total) %>% .[1]
      x <- unique(df$x)

      mu_u_t <-  diag(K) %*% solve(B %*% t(B) + lambda) %*% B
      sigma_u_t  <- diag(M) - t(B) %*% solve(B %*% t(B) + lambda) %*% B
      sigma_u_t_chol <- solve(chol(sigma_u_t))
      bias_max <- sigma_total * apply(mu_u_t %*% sigma_u_t_chol, 1, function(x) sqrt(sum(x^2)))[1] * abs(x)
      bias_random  <- sigma_total * apply(mu_u_t %*% sigma_u_t_chol %*% rstiefel::rustiefel(M, 1), 1, function(x) sqrt(sum(x^2)))[1] * abs(x)
      yhat_median <- df %>% filter(x == 0) %>% pull(yhat) %>% .[1]

      yhats <- df %>% select(x, gene, yhat) %>% distinct %>% mutate(diff = yhat - yhat_median)

      r2 <- runif(1, 0, 1)

      tibble(x = rep(yhats$x, 2),
             y0 = rep(yhats$diff, 2),
             y25 = c(yhats$diff + sqrt(0.25) * bias_max, yhats$diff - sqrt(0.25) * bias_max),
             y50 = c(yhats$diff + sqrt(0.50) * bias_max, yhats$diff - sqrt(0.5) * bias_max),
             y75 = c(yhats$diff + sqrt(0.75) * bias_max, yhats$diff - sqrt(0.75) * bias_max),
             y100 = c(yhats$diff + bias_max, yhats$diff - bias_max),
             y100_dunif = c(yhats$diff + bias_random, yhats$diff - bias_random),
             y100_dunif_r2unif = c(yhats$diff + sqrt(r2) * bias_random, yhats$diff - sqrt(r2) * bias_random))%>%
        pivot_longer(cols=y0:y100_dunif_r2unif, names_to="R2")

    })) %>%
  unnest(cols=c(bias))


  bias_tibble_r2 <- bias_tibble %>% filter(!(R2 %in% c("y100_dunif", "y100_dunif_r2unif")))

  bias_tibble_r2 %>%
    group_by(x, R2) %>% summarize(lower=quantile(value, 0.025), upper=quantile(value, 0.975)) %>%
    ungroup() %>%
    mutate(R2 = fct_recode(R2, `0.00`="y0", `0.25`="y25", `0.50`="y50", `0.75`="y75", `1.00`="y100")) %>%
    mutate(R2 = fct_rev(fct_relevel(R2, c("0.00", "0.25", "0.50", "0.75", "1.00")))) %>%
    ggplot() +
    geom_ribbon(aes(x=x, ymin=lower, ymax=upper, fill=R2), alpha=0.75) + theme_bw(base_size=24) +
    geom_line(data=bias_tibble %>% group_by(x) %>% summarize(pm = mean(value)), aes(x=x, y=pm), linetype="dashed") +
    colorspace::scale_fill_discrete_sequential("Emrld", rev=FALSE) +
    geom_hline(yintercept=0, linetype="dashed", size=1.2, col="firebrick1") +
    ylab(expression(tau[x])) + xlab("X") + ggtitle(sprintf("Treatment %i", gene)) +
    theme(axis.title.y = element_text(angle = 0, vjust = 0.5, hjust=1, size=30)) +
    guides(fill=guide_legend(title=expression(R[paste(Y, '~', U, '|', T)]^2))) + ylim(c(-10, 10))

  ggsave(file=sprintf("../figs/bart_simulation_%i.pdf", gene), width=10)
}

####################################
##  Null Controls
###################################33

### Bias
samples_joined  <-  full_join(samples_tibble,
                              factor_samples, by=c(".draw"))

bias_tibble <- samples_joined %>%
  group_by(`.draw`) %>%
  nest() %>%
  transmute(bias = map(data, function(df) {

    ## Null control
    df1 <- df %>% filter(gene == 1)
    B <- df1 %>% filter(x == -2) %>% pull(B) %>% matrix(., nrow=K, byrow=TRUE)
    lambda <- df1 %>%  filter(x == -2, M==1) %>% pull(lambda) %>% diag
    sigma_total  <- df1 %>% pull(sigma_total) %>% .[1]
    K <- max(df1$K)


    mu_u_t_null <-  c(2, rep(0, K-1)) %*% solve(B %*% t(B) + lambda) %*% B
    sigma_u_t  <- diag(M) - t(B) %*% solve(B %*% t(B) + lambda) %*% B
    sigma_u_t_chol_inv <- solve(chol(sigma_u_t))

    ## -2 vs 0
    tau_naive <- df1 %>% filter(gene==1, x %in% c(-2, 0), M==1, K==1) %>%
      arrange(desc(x)) %>%
      pull(yhat) %>% diff

    M  <- sigma_u_t_chol_inv %*% t(mu_u_t_null)

    tau_M <- tau_naive %*% MASS::ginv(M)
    R2min  <- sum(tau_M^2) / sigma_total^2

    treatment_bias <- tau_naive %*% MASS::ginv(M) %*% sigma_u_t_chol_inv %*% t(diag(K) %*% solve(B %*% t(B) + lambda) %*% B) %>% as.vector
    
    Pperp  <- NullC(sigma_u_t_chol_inv %*% t(mu_u_t_null))

    post_null_bias_direction <- Pperp %*% t(Pperp) %*% sigma_u_t_chol_inv %*% t(diag(K) %*% solve(B %*% t(B) + lambda) %*% B)
    post_null_bias_norm <- sigma_total * apply(post_null_bias_direction, 2, function(x) sqrt(sum(x^2)))

    yhat_zero <- df %>% filter(x == 0) %>% select(x, gene, yhat) %>% distinct %>% select(gene, yhat) %>% rename(yzero=yhat)
    yhats <- df %>% select(x, gene, yhat) %>% distinct %>% left_join(., yhat_zero, by="gene") %>% mutate(diff = yhat - yzero)
    yhats <- left_join(yhats, tibble(gene=1:K, scaled_bias=treatment_bias, scaled_interval_width=post_null_bias_norm), by="gene")
    yhats$bias <- yhats$scaled_bias * yhats$x
    yhats$interval_width <- yhats$scaled_interval_width * yhats$x


    tibble(
      gene = rep(yhats$gene, 2),
      x = rep(yhats$x, 2),
      R2min = R2min,
      y0 = rep(yhats$diff + yhats$bias, 2),
      y25 = c(yhats$diff + yhats$bias + sqrt(max(0.25 - R2min, 0)) * yhats$interval_width, yhats$diff + yhats$bias - sqrt(max(0.25 - R2min, 0)) * yhats$interval_width),
      y50 = c(yhats$diff + yhats$bias + sqrt(max(0.50 - R2min, 0)) * yhats$interval_width, yhats$diff + yhats$bias - sqrt(max(0.50 - R2min, 0)) * yhats$interval_width),
      y75 = c(yhats$diff + yhats$bias + sqrt(max(0.75 - R2min, 0)) * yhats$interval_width, yhats$diff + yhats$bias - sqrt(max(0.75 - R2min, 0)) * yhats$interval_width),
      y100 = c(yhats$diff + yhats$bias + sqrt(max(1 - R2min, 0)) * yhats$interval_width, yhats$diff + yhats$bias - sqrt(max(1 - R2min, 0)) * yhats$interval_width)
    ) %>%
      pivot_longer(cols=y0:y100, names_to="R2")
  })) %>%
  unnest(cols=c(bias))

## Plot gene 1 corrected
for(gene in c(1, 2, 6)) {

  gene_bias <- bias_tibble %>%
    filter(gene == !!gene) %>%
    mutate(R2 = factor(R2)) %>%
    filter(R2 != "y0",
           !(R2min > 0.25 & R2 == "y25"),
           !(R2min > 0.5 & R2 == "y50"),
           !(R2min > 0.75 & R2 == "y75"))

  gene_bias %>% 
  group_by(x, R2) %>% summarize(lower=quantile(value, 0.025), upper=quantile(value, 0.975)) %>%
  ungroup() %>%
  mutate(R2 = fct_recode(R2, `0.00`="y0", `0.25`="y25", `0.50`="y50", `0.75`="y75", `1.00`="y100")) %>%
  mutate(R2 = fct_rev(fct_relevel(R2, c("0.00", "0.25", "0.50", "0.75", "1.00")))) %>%
  ggplot() +
  geom_ribbon(aes(x=x, ymin=lower, ymax=upper, fill=R2), alpha=0.75) + theme_bw(base_size=24) +
  geom_line(data=gene_bias %>% group_by(x) %>% summarize(pm = mean(value)), aes(x=x, y=pm), linetype="dashed") +
  ## colorspace::scale_fill_discrete_sequential("Emrld", rev=FALSE) +
  scale_fill_manual(values=sequential_hcl(5, "Emrld")[1:4]) +
  geom_hline(yintercept=0, linetype="dashed", size=1.2, col="firebrick1") +
  ylab(expression(tau[x])) + xlab("X") + ggtitle(sprintf("Treatment %i (1 is NC)", gene)) +
  theme(axis.title.y = element_text(angle = 0, vjust = 0.5, hjust=1, size=30)) +
  guides(fill=guide_legend(title=expression(R[paste(Y, '~', U, '|', T)]^2))) + ylim(c(-10, 10))

ggsave(file=sprintf("../figs/bart_simulation_null_%i.pdf", gene), width=10)
}


## Histogrma of R2min
bias_tibble %>% filter(gene == 1) %>% ggplot() +
  geom_histogram(aes(x=R2min)) +
  #geom_vline(xintercept=0.5, color="firebrick", linetype="dashed")  +
  theme_bw(base_size=16) + xlim(c(0, 1)) +
  labs(title = "Distribution of Minimum R-squared",
              subtitle = "Null Control Assumption on Treatment 1")
ggsave(file="../figs/bart_min_r2.pdf")
