// Fit the hyperparameters of a latent-variable Gaussian process with an
// ARD-parameterized exponentiated quadratic kernel and a Gaussian likelihood
// https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html
// https://arxiv.org/pdf/1609.00046.pdf
// http://mc-stan.org/rstanarm/reference/priors.html
// http://mc-stan.org/rstanarm/articles/priors.html

functions {
  matrix L_cov_exp_quad_ARD(vector[] x,
                            real alpha,
                            vector rho,
                            real delta) {
    int N = size(x);
    matrix[N, N] K;
    real neg_half = -0.5;
    real sq_alpha = square(alpha);
    for (i in 1:(N-1)) {
      K[i, i] = sq_alpha + delta;
      for (j in (i + 1):N) {
        K[i, j] = sq_alpha * exp(neg_half * 
                                 dot_self((x[i] - x[j]) ./ rho));
        K[j, i] = K[i, j]; 
      }
    }
    K[N, N] = sq_alpha + delta;
    return cholesky_decompose(K);
  }
}
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=0> Npred;
  int<lower=1> M; // Number of confounders

  vector[K] X[N];
  vector[K] Xpred[Npred];
  vector[N] y;
  
}
transformed data {
  real delta = 1e-9;
  int Ntotal = N + Npred;
  vector[K] Xtotal[N+Npred];
  for(n in 1:N) {
   Xtotal[n] = X[n];
  }
  for(n in 1:Npred) {
   Xtotal[N + n] = Xpred[n];
  }
}
parameters {
  // regression parameters
  real intercept;
  vector[K] beta;

  //GP parameters
  vector<lower=0>[K] rho;
  real<lower=0> alpha;
  real<lower=0> sigma_total;
  vector[Ntotal] eta;
  
  // confounders
  matrix[K, M] B; //
  vector<lower=0>[K] D;  
  unit_vector[M] d;
  real<lower=0, upper=1> r2;

} transformed parameters{

  cov_matrix[K] sigma_X_inv = diag_matrix(rep_vector(1, K)./D) - diag_matrix(rep_vector(1, K)./D) * B * inverse_spd(diag_matrix(rep_vector(1, M)) + B' * diag_matrix(rep_vector(1, K)./D) * B) * B' * diag_matrix(rep_vector(1, K)./D);
  cov_matrix[M] sigma_u_t = diag_matrix(rep_vector(1, M)) - B' * sigma_X_inv * B;
  cholesky_factor_cov[M] sigma_u_t_root_inv = cholesky_decompose(inverse_spd(sigma_u_t));
  vector[M] gamma = sigma_total*sqrt(r2)*sigma_u_t_root_inv*d;
  vector[K] bias = sigma_X_inv * B * gamma;
  vector[Ntotal] f =  L_cov_exp_quad_ARD(Xtotal, alpha, rho, delta) * eta;   
}
model {
  
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(0, 1);
  sigma_total ~ normal(0, 1);
  eta ~ normal(0, 1);
  for(n in 1:N) {
     X[n] ~  multi_normal_prec(rep_vector(0, K), sigma_X_inv);
     y[n] ~ normal(intercept + (beta + bias)' * Xtotal[n] + f[n], sigma_total);
  }
  
} generated quantities {
  vector[N] yppc;
  vector[Npred] ypred;
  vector[N] mu_obs;
  vector[Npred] mu_pred;
  for(n in 1:N) {
    mu_obs[n] = intercept + (beta + bias)' * X[n] + f[n];
    yppc[n] = normal_rng(mu_obs[n], sigma_total);
  }
  for(n in 1:Npred) {
    mu_pred[n] = intercept + (beta + bias)' * Xpred[n] + f[N+n];
    ypred[n] = normal_rng(mu_pred[n], sigma_total);
  }
}
