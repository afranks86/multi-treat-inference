// https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html

data {

  int<lower=1> N; // Number of data points
  int<lower=1> K; // Number of covariates
  int<lower=1> M; // Number of confounders

  // regularizing parameters
  real<lower=0, upper=1> frac_non_null;
  real slab_scale;    // Scale for large slopes
  
  matrix[N, K] X;
  real y[N];

}
transformed data {
  real k0 = frac_non_null*K;           // Expected fraction of large slopes is 0.2
  real slab_scale2 = square(slab_scale);
  real slab_df = 25;      // Effective degrees of freedom for large slopes
  real half_slab_df = 0.5 * slab_df;
}
parameters {

  // Finnish horseshoe params
  vector[K] beta_tilde;
  vector<lower=0>[K] lambda;
  real<lower=0> c2_tilde;
  real<lower=0> tau_tilde;

  // other params
  matrix[K, M] B; //
  real alpha;
  real<lower=0> sigma_total;
  real<lower=0> sigma_treat;
  unit_vector[M] d;
  real<lower=0, upper=1> r2;



}
transformed parameters {

  cov_matrix[K] sigma_X_inv = diag_matrix(rep_vector(1.0 / sigma_treat^2, K)) - 1.0 /sigma_treat^4 * B * inverse_spd(diag_matrix(rep_vector(1, M)) + 1/sigma_treat^2 * B' * B) * B';
  cov_matrix[M] sigma_u_t = diag_matrix(rep_vector(1, M)) - B' * sigma_X_inv * B;
  cholesky_factor_cov[M] sigma_u_t_root_inv = cholesky_decompose(inverse_spd(sigma_u_t));
  vector[M] gamma = sigma_total*sqrt(r2)*sigma_u_t_root_inv*d;
  vector[K] bias = sigma_X_inv * B * gamma;


  // For the Finnish Horseshoe
  vector[K] beta;
  {
    real tau0 = (k0 / (K - k0)) * (sigma_total / sqrt(1.0 * N));
    real tau = tau0 * tau_tilde; // tau ~ cauchy(0, tau0)

    // c2 ~ inv_gamma(half_slab_df, half_slab_df * slab_scale2)
    // Implies that marginally beta ~ student_t(slab_df, 0, slab_scale)
    real c2 = slab_scale2 * c2_tilde;

    vector[K] lambda_tilde =
      sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );

    // beta ~ normal(0, tau * lambda_tilde)
    beta = tau * lambda_tilde .* beta_tilde;
  }
  
}
model {
  
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);

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
