// https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html
// https://arxiv.org/pdf/1609.00046.pdf
// http://mc-stan.org/rstanarm/reference/priors.html
// http://mc-stan.org/rstanarm/articles/priors.html

data {

  int<lower=1> N; // Number of data points
  int<lower=1> K; // Number of covariates
  int<lower=1> M; // Number of confounders

  matrix[N, K] X;
  real y[N];
  
  int<lower=0, upper=M> num_null;
  int<lower=1, upper=K> null_control_indices[num_null];
  int<lower=1, upper=K> non_null_control_indices[K - num_null];
}
parameters {
  vector[K - num_null] beta_free;
  matrix[K, M] B; //
  real alpha;
  real<lower=0> sigma_total;
  real<lower=0> sigma_treat;
  unit_vector[M] d;
  real<lower=0, upper=1> r2;

}
transformed parameters {
  vector[K] beta;
  
  cov_matrix[K] sigma_X_inv = diag_matrix(rep_vector(1.0 / sigma_treat^2, K)) - 1.0 /sigma_treat^4 * B * inverse(diag_matrix(rep_vector(1, M)) + 1/sigma_treat^2 * B' * B) * B';
  cov_matrix[M] sigma_u_t = diag_matrix(rep_vector(1, M)) - B' * sigma_X_inv * B;
  cholesky_factor_cov[M] sigma_u_t_root_inv = cholesky_decompose(inverse(sigma_u_t));
  vector[M] gamma = sigma_total*sqrt(r2)*sigma_u_t_root_inv*d;
  vector[K] bias = sigma_X_inv * B * gamma;

  beta[null_control_indices] = rep_vector(0.0, num_null);
  beta[non_null_control_indices] = beta_free;
}
model {

  for(n in 1:N){
    X[n] ~ multi_normal_prec(rep_vector(0, K), sigma_X_inv);
    y[n] ~ normal(alpha + X[n] * (beta + bias), sigma_total);
  }

} generated quantities {
  vector[N] log_lik;
  for(n in 1:N) {
    log_lik[n] = multi_normal_prec_lpdf(X[n] | rep_vector(0, K), sigma_X_inv) + normal_lpdf(y[n] | alpha + X[n] * (beta + bias), sigma_total);
  }
}
