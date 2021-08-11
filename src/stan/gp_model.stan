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
  int<lower=1> D;
  int<lower=0> Npred;

  vector[D] X[N];
  vector[D] Xpred[Npred];
  vector[N] y;
}
transformed data {
  real delta = 1e-9;
  int Ntotal = N + Npred;
  vector[D] Xtotal[N+Npred];
  for(n in 1:N) {
   Xtotal[n] = X[n];
  }
  for(n in 1:Npred) {
   Xtotal[N + n] = Xpred[n];
  }
}
parameters {
  real intercept;
  vector[D] beta;
  vector<lower=0>[D] rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
  vector[Ntotal] eta;

} transformed parameters{

   vector[Ntotal] f;   
   matrix[Ntotal, Ntotal] L_K = L_cov_exp_quad_ARD(Xtotal, alpha, rho, delta);
   f = L_K * eta;
}
model {
  

  
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);
  eta ~ normal(0, 1);
  for(n in 1:N) {
     y[n] ~ normal(intercept + beta' * Xtotal[n] + f[n], sigma);
  }
} generated quantities {
  vector[N] yppc;
  vector[Npred] ypred;
  vector[N] mu_obs;
  vector[Npred] mu_pred;
  for(n in 1:N) {
    mu_obs[n] = intercept + beta' * X[n] + f[n];
    yppc[n] = normal_rng(mu_obs[n], sigma);
  }
  for(n in 1:Npred) {
    mu_pred[n] = intercept + beta' * Xpred[n] + f[N+n];
    ypred[n] = normal_rng(mu_pred[n], sigma);
  }
}
