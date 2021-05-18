// https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html

data {

  int<lower=1> N; // Number of data points
  int<lower=1> K; // Number of covariates
  int<lower=1> M; // Number of confounders

  matrix[N, K] X;
  real y[N];

} 
parameters {

  vector[K] beta;
  matrix[K, M] B; //
  real alpha;
  real<lower=0> sigma_y;
  real<lower=0> sigma_treat;
  vector[M] gamma;


}
transformed parameters {
  cov_matrix[K] sigma_X_inv = diag_matrix(rep_vector(1.0 / sigma_treat^2, K)) - 1.0 /sigma_treat^4 * B * inverse(diag_matrix(rep_vector(1, M)) + 1/sigma_treat^2 * B' * B) * B';
  // cov_matrix[K] sigma_X_inv = inverse(diag_matrix(rep_vector(sigma_treat^2, K)) + B * B');
  cov_matrix[M] sigma_u_t = diag_matrix(rep_vector(1, M)) - B' * sigma_X_inv * B;
  real sigma_total = sqrt(sigma_y^2 +  gamma' * sigma_u_t * gamma);
  vector[K] bias = sigma_X_inv * B * gamma;
}
model {

  for(n in 1:N){
    X[n] ~ multi_normal_prec(rep_vector(0, K), sigma_X_inv);
    y[n] ~ normal(alpha + X[n] * (beta + bias), sigma_total);
  }

}
generated quantities{
  real r2 = 1 - (sigma_y^2 / sigma_total^2);
}
