// https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html
// https://arxiv.org/pdf/1609.00046.pdf
// http://mc-stan.org/rstanarm/reference/priors.html
// http://mc-stan.org/rstanarm/articles/priors.html

data {

  int<lower=1> N; // Number of data points
  int<lower=1> K; // Number of covariates
  int<lower=1> M; // Number of confounders

  matrix[N, K] X;

}
parameters {

  matrix[K, M] B; //
  vector<lower=0>[K] lambda;

}
transformed parameters {
  //cov_matrix[K] sigma_X_inv = diag_matrix(rep_vector(1.0 / sigma_treat^2, K)) - 1.0 /sigma_treat^4 * B * inverse_spd(diag_matrix(rep_vector(1, M)) + 1/sigma_treat^2 * B' * B) * B';
  matrix[K, K] lambda_inv = diag_matrix(1.0 ./ lambda);
  cov_matrix[K] sigma_X_inv = lambda_inv - lambda_inv * B * inverse_spd(diag_matrix(rep_vector(1, M)) +  B' * lambda_inv * B) * B' * lambda_inv;
}
model {

  for(n in 1:N){
    X[n] ~ multi_normal_prec(rep_vector(0, K), sigma_X_inv);
  }

} generated quantities {

  vector[N] log_lik;
  for(n in 1:N) {
    log_lik[n] = multi_normal_prec_lpdf(X[n] | rep_vector(0, K), sigma_X_inv);
  }

}
