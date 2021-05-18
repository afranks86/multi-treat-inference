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
  real<lower=0> sigma_beta;

  real<lower=0> alpha_scaler;
  real<lower=0> beta_scaler;

} transformed data{
  real my = mean(y);
  real<lower=0> sdy = sd(y);
  real<lower=0> sdx[K];
  for(k in 1:K){
    sdx[k] = sd((X')[k]);
  }
}
parameters {
  vector[K] beta;
  matrix[K, M] B; //
  real alpha;
  real<lower=0> sigma_total;
  real<lower=0> sigma_treat;
  unit_vector[M] d;
  real<lower=0, upper=1> r2;

}
transformed parameters {
  cov_matrix[K] sigma_X_inv = diag_matrix(rep_vector(1.0 / sigma_treat^2, K)) - 1.0 /sigma_treat^4 * B * inverse(diag_matrix(rep_vector(1, M)) + 1/sigma_treat^2 * B' * B) * B';
  cov_matrix[M] sigma_u_t = diag_matrix(rep_vector(1, M)) - B' * sigma_X_inv * B;
  cholesky_factor_cov[M] sigma_u_t_root_inv = cholesky_decompose(inverse(sigma_u_t));
  vector[M] gamma = sigma_total*sqrt(r2)*sigma_u_t_root_inv*d;
  vector[K] bias = sigma_X_inv * B * gamma;
}
model {

  sigma_total ~ exponential(1 / sdy);
  for(k in 1:K) {
    beta[k] ~ normal(0, beta_scaler*2.5*sdy/sdx[k])
  }
  alpha ~ normal(my, alpha_scaler*2.5*sdy)

  for(n in 1:N){
    X[n] ~ multi_normal_prec(rep_vector(0, K), sigma_X_inv);
    y[n] ~ normal(alpha + X[n] * (beta + bias), sigma_total);
  }

}
