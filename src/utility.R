generate_sparse_data  <- function(n, k, m, sparsity, seed=NULL) {

  if(is.null(seed))
    seed  <- sample(1e6, 1)
  set.seed(seed)

  u <- matrix(rnorm(n*m), ncol=m)

  B  <- matrix(rnorm(k*m, sd=2), ncol=k)

  sigma_t  <- 1
  t <- u %*% B + matrix(rnorm(k*n), ncol=k)

  tau  <- ifelse(runif(k) < sparsity, rnorm(k, sd=2), rep(0, k))

  Sigma_t  <- t(B) %*% B + diag(sigma_t^2, k)
  Sigma_inv  <- solve(t(B) %*% B + diag(sigma_t^2, k))
  r2t  <- t(B) %*% B %*% Sigma_inv
  Sigma_u_t  <- diag(m) - B %*%  Sigma_inv %*% t(B)

  gamma  <- 50*eigen(Sigma_u_t)$vectors[, 1]
  bias  <- t(gamma) %*% B %*% Sigma_inv  %*% diag(k)

  var_u  <-  t(gamma) %*% Sigma_u_t %*% gamma


  sigma_y <- 1
  r2y  <- var_u / (var_u + sigma_y^2)

  y  <- t %*% tau + u %*% gamma + rnorm(n, 0, sigma_y)

  sigma_total  <- sqrt(var_u + sigma_y^2)

  return(list(n=n, k=k, m=m, y=y, t=t, u=u, tau=tau, bias=bias,
              sigma_y=sigma_y, sigma_t=sigma_t, sigma_total=sigma_total,
              gamma = gamma, Sigma_u_t = Sigma_u_t,
              r2t=r2t, r2y=r2y, B=B, seed=seed))

}


generate_data_null  <- function(n, k, m, null_treatments=TRUE, r2=0.5, seed=NULL) {

  normalize  <- function(x) { x / sqrt(sum(x^2)) }

  if(null_treatments) {
    tau <- rep(0, k)
    obs_tau <- rep(c(1, -1), each= k/2)
  } else {
    obs_tau <- rep(c(1, -1), each= k/2)
    tau <- 2*obs_tau
  }

  b1  <- normalize(tau - obs_tau)
 
  sigma_t  <- 1
  Sigma_u_t  <- diag(0.2, m)

  diag_mat <- diag(sqrt((1-diag(Sigma_u_t)) / (diag(Sigma_u_t))), nrow=m, ncol=m)
  if(m == 1) {
    B  <- t(b1 %*% diag_mat)
  } else if (m == 2) {
    b2  <- normalize(rep(1, k))
    B  <- t(cbind(b1, b2) %*% diag_mat)
  } else {
    b2  <- normalize(rep(1, k))
    B  <- t(cbind(b1, b2, rstiefel::NullC(cbind(b1, b2))[, 1:(m-2)]) %*% diag_mat)
  }

  Sigma_inv  <- solve(t(B) %*% B + diag(sigma_t^2, k))

  bias  <- obs_tau - tau
  gamma <- t(bias %*% MASS::ginv(B %*% Sigma_inv  %*% diag(k)))

  Sigma_u_t  <- diag(m) - B %*% Sigma_inv %*% t(B)

  sigma_u  <-  sqrt(t(gamma) %*% Sigma_u_t %*% gamma)
  sigma_total  <- sqrt(sigma_u^2 / r2)

  computed_bias  <- as.numeric(t(gamma) %*% B %*% Sigma_inv  %*% diag(k))
  if(!isTRUE(all.equal(computed_bias, bias))) {
    gamma <- -1 * gamma
    if(!isTRUE(all.equal(as.numeric(t(gamma) %*% B %*% Sigma_inv  %*% diag(k)), bias)))
      stop("Error in data generator")
  }

  sigma_y  <- sqrt(sigma_total^2 - sigma_u^2)

  r2t  <- t(B) %*% B %*% Sigma_inv
  r2y  <- r2
  
  if(is.null(seed))
    seed  <- sample(1e6, 1)
  set.seed(seed)

  u <- matrix(rnorm(n*m), ncol=m)
  t <- u %*% B + matrix(rnorm(k*n, sd=sigma_t), ncol=k)
  y  <- t %*% tau + u %*% gamma + rnorm(n, 0, sigma_y)

  return(list(n=n, k=k, m=m, y=y, t=t, u=u, tau=tau, obs_tau=obs_tau, bias=bias,
              sigma_y=sigma_y, sigma_t=sigma_t, sigma_total=sigma_total,
              gamma = gamma, Sigma_u_t = Sigma_u_t,
              r2t=r2t, r2y=r2y, B=B, seed=seed))


}

plot_gammas  <- function(res, filters, facet_cols = 5, facet_vars=c("model"), circle_radius=0, color_var=sym("RMSE")) {

  circle_dat  <- res %>% ungroup %>% filters %>% select(model, ntreat, n, r) %>%  distinct() %>% mutate(zero=0)
  as_tibble(res) %>% filters %>% select(M, gamma, r2, !!color_var, n, ntreat, model) %>% distinct() %>%
    arrange(desc(!!color_var)) %>%
    pivot_wider(names_from=M, values_from=gamma, names_prefix="g") %>%
    ggplot() +
    geom_point(aes(x=g1, y=g2, col=!!color_var), alpha=0.75, size=0.5) + theme_bw(base_size=16) +
    scale_color_continuous_sequential(palette = "Viridis")  +
    ylab("Gamma[2]") + xlab("Gamma[1]") + 
    ggforce::geom_circle(data=circle_dat, aes(x0=zero, y0=zero, r=r)) +
    facet_wrap(vars(!!!syms(facet_vars)), ncol=facet_cols, labeller = label_wrap_gen(multi_line=FALSE))

}

## R^2 Histogram
plot_r2_histograms  <- function(res, filters = function(x) {x}, facet_cols = 5, facet_vars=c("model")) {


  as_tibble(res) %>% filters %>% select(M, m, gamma, r2, RMSE, n, ntreat, model) %>% distinct() %>%
    arrange(desc(RMSE)) %>%
    pivot_wider(names_from=M, values_from=gamma, names_prefix="g") %>%
    ggplot() + geom_histogram(aes(x=r2, col=RMSE), alpha=0.75) + theme_bw(base_size=16) +
    facet_wrap(vars(!!!syms(facet_vars)), ncol=facet_cols, labeller = label_wrap_gen(multi_line=FALSE)) +
    scale_color_continuous_sequential(palette = "Viridis") + ylab("Count")  + xlab("Partial Variance Explained by Confounders")

}
