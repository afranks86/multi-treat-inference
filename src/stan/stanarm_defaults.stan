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
  matrix[N, M] U; // unobserved confounders
  real alpha;
  real<lower=0> sigma_treat;
  real<lower=0> sigma_y;
  vector[M] gamma;

} transformed parameters {
  cov_matrix[K] sigma_X_inv = diag_matrix(rep_vector(1.0 / sigma_treat^2, K)) - 1.0 /sigma_treat^4 * B * inverse(diag_matrix(rep_vector(1, M)) + 1/sigma_treat^2 * B' * B) * B';
}
model {

  sigma_y ~ exponential(1 / sdy);
  for(k in 1:K) {
    beta[k] ~ normal(0, beta_scaler*2.5*sdy/sdx[k]);
  } 
  for(m in 1:M) {
    gamma[m] ~ normal(0, beta_scaler*2.5*sdy/sd((U')[m]));
  }

  alpha ~ normal(my, alpha_scaler*2.5*sdy);

  for(n in 1:N){
    U[n] ~ normal(0, 1);
    X[n] ~ multi_normal(U[n] * B', diag_matrix(square(rep_vector(sigma_treat, K))));
    y[n] ~ normal(alpha + X[n] * beta + U[n] * gamma, sigma_y);
  }

} generated quantities{
  real<lower=0> sigma_total = sqrt(sigma_y^2 + gamma' * B' * sigma_X_inv * B * gamma);
  real r2 = 1 - (sigma_y^2 / sigma_total^2);
  vector[K] bias = sigma_X_inv * B * gamma;
}
