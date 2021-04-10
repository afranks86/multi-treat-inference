generate_sparse_data  <- function(n, k, m, sparsity, seed=NULL) {

  if(is.null(seed))
    seed  <- sample(1e6, 1)
  set.seed(seed)

  u <- matrix(rnorm(n*m), ncol=m)

  B  <- matrix(rnorm(k*m, sd=2), ncol=k)

  sigma_t  <- 1
  t <- u %*% B + matrix(rnorm(k*n), ncol=k)

  tau  <- ifelse(runif(k) < sparsity, rnorm(k, sd=2), 0)

  Sigma_t  <- t(B) %*% B + diag(sigma_t^2, k)
  Sigma_inv  <- solve(t(B) %*% B + diag(sigma_t^2, k))
  r2t  <- t(B) %*% B %*% Sigma_inv
  Sigma_u_t  <- diag(m) - B %*%  Sigma_inv %*% t(B)

  gamma  <- 50*eigen(Sigma_u_t)$vectors[, 1]
  bias  <- t(gamma) %*% B %*% Sigma_inv  %*% diag(k)

  var_u  <-  t(gamma) %*% Sigma_u_t %*% gamma


  sigma_y <- 1
  r2y  <- var_u / (var_u + sigma_y^2)
  
  y  <- t %*% tau + u %*% gamma + rnorm(n, sigma_y)

  sigma_total  <- sqrt(var_u + sigma_y^2)

  return(list(y=y, t=t, u=u, tau=tau, bias=bias,
              sigma_y=sigma_y, sigma_t=sigma_t, sigma_total=sigma_total,
              gamma = gamma, Sigma_u_t = Sigma_u_t,
              r2t=r2t, r2y=r2y, B=B, seed=seed))

}
