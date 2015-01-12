# basic model but estimating tridiagonal covariance matrix

functions {
  # mean without error correction term
  real mean_wo_correction(real D_0, real rho, vector beta, 
                          real party_L1, real util, real econ) {
    real val ;
    val <- D_0 + rho * party_L1 + 
      (1 - rho) * (beta[1] + beta[2] * util + beta[3] * econ) ;
    return val ;
  }
  
  # error correction term
  real error_correction(real D_0, real rho_t1, vector beta, 
                        real party_L1, real party_L2, 
                        real util_L1, real econ_L1) {
    real val ; 
    val <- party_L1 - D_0 - 
      rho_t1 * party_L2 - 
      (1-rho_t1) * (beta[1] + beta[2]*util_L1 + beta[3]*econ_L1) ;
    return val ;
  }
  
  # mean with error correction term
  real mean_w_correction(real delta, real D_0, real rho, real rho_t1, vector beta,
                         real party_L1, real party_L2, real util, real util_L1,
                         real econ, real econ_L1) {
    
    real m ; 
    real V ;
    
    m <- mean_wo_correction(D_0, rho, beta, party_L1, util, econ) ;
    V <- error_correction(D_0, rho_t1, beta, party_L1, party_L2, util_L1, econ_L1) ;
    return m - delta*V ;
  }
  
  # increment deviance
  real increment_deviance(real dev, real y, real mu, real sigma) {
    real new_term ;
    new_term <- (-2) * normal_log(y, mu, sigma) ;
    return dev + new_term ; 
  }
  
  vector inv_logit_vec(vector x) {
    vector[num_elements(x)] out ;
    for (j in 1:num_elements(out)) {
      out[j] <- inv_logit(x[j]) ;
    }
    return out ;
  }
  
  real cauchy_trans(real loc, real scale, real noise) {
    return loc + scale * tan(pi() * (Phi_approx(noise) - 0.5)) ;
  }
  
  vector cauchy_trans_vec(vector loc, vector scale, vector noise) {
    vector[num_elements(noise)] out ;
    for (j in 1:num_elements(out)) {
      out[j] <- loc[j] + scale[j] * tan(pi() * (Phi_approx(noise[j]) - 0.5)) ;
    }
    return out ;
  }
}

data {
  int<lower=1>                  T ; # number of time periods
  int<lower=1>                  J ; # number of groups
  matrix[T,J]                   party ; # mean partyid 
  matrix[T,J]                   party_L1 ; # party, 1 lag
  matrix[T,J]                   party_L2 ; # party, 2 lags
  matrix[T,J]                   util ;
  matrix[T,J]                   util_L1 ;
  vector[T]                     econ ; # retrospective economic evaluation
  vector[T]                     econ_L1 ; # E, 1 lag  
}

transformed data {
  vector[T]       gp_mu ;
  gp_mu <- rep_vector(0, T) ;
}

parameters {  
  vector<lower=0>[T+2]      gp_noise ;
  vector[T]                 gp_z ;
  real<lower=0>             sigma_noise ; 
  vector[3]                 beta_noise ; 
  real                      D_0 ; # intercept
  real                      delta_noise ; # coeff on error correction term
}

transformed parameters {
  vector<lower=0>[T] gp_tau ;
  real<lower=0>   gp_nu ;
  real<lower=0>   gp_eta ;
  matrix[T,T]     gp_L ;
  vector[T]       rho ;
  vector[3]       beta ;
  real            delta ;
  real<lower=0>   sigma ;
  
  delta   <- 2.5 * delta_noise ;
  beta    <- 5.0 * beta_noise ; 
  sigma   <- cauchy_trans(0, 2.5, sigma_noise) ;
  gp_nu   <- cauchy_trans(0, 1, gp_noise[T+1]) ;
  gp_eta  <- cauchy_trans(0, 1, gp_noise[T+2]) ;
  gp_tau  <- cauchy_trans_vec(rep_vector(0,T), rep_vector(1,T), segment(gp_noise,1,T)) ;
  
  { # local
  matrix[T,T]   gp_Sigma ;
  for (t1 in 1:T) {
    for (t2 in 1:T) {
      if (t1 == t2) gp_Sigma[t1, t2] <- gp_tau[t1] ;
      else 
        if (fabs(t1-t2) == 1) gp_Sigma[t1, t2] <- gp_nu ;
        else 
          if (fabs(t1-t2) == 2) gp_Sigma[t1, t2] <- gp_eta ;
          else gp_Sigma[t1, t2] <- 0.0 ;
    }
  }
  
  gp_L <- cholesky_decompose(gp_Sigma) ;
  } # end local
  
  rho <- inv_logit_vec(gp_mu + gp_L * gp_z) ;
}

model {
  # Priors
  beta_noise ~ normal(0, 1) ; # implies beta[i] ~ normal(beta_mean, beta_sd)
  D_0 ~ normal(0, 1) ;
  delta_noise ~ normal(0, 1) ;
  sigma_noise ~ normal(0, 1) ;
  gp_noise ~ normal(0, 1) ;
  gp_z ~ normal(0, 1) ; 
  
  
  # Likelihood
  for (j in 1:J) {
    vector[T] mu ;
    mu[1] <- mean_wo_correction(D_0, rho[1], beta, party_L1[1,j], util[1,j], econ[1]) ;
    for(t in 2:T) mu[t] <- mean_w_correction(delta, D_0, rho[t], rho[t-1], beta,
                                             party_L1[t,j], party_L2[t,j], 
                                             util[t,j], util_L1[t,j],
                                             econ[t], econ_L1[t]) ;
    
    col(party, j) ~ normal(mu, sigma) ;      
  }
  
}

generated quantities {
  real dev ; # deviance
  matrix[T,J] party_rep ; # simulated values
  matrix[T,J] resids_rep ; # residuals: party[t,j] - party_rep[t,j]
  vector[T*J] log_lik ; # log likelihood (for computing waic)
  
  { # local 
  matrix[T,J] log_lik_mat ;
  
  dev <- 0.0 ;
  for (j in 1:J) {
    for(t in 1:T) {
      real mu_post ;
      
      if (t == 1) 
        mu_post <- mean_wo_correction(D_0, rho[t], beta, party_L1[t,j], util[t,j], econ[t]) ;
      else 
        mu_post <- mean_w_correction(delta, D_0, rho[t], rho[t-1], beta,
                                     party_L1[t,j], party_L2[t,j], 
                                     util[t,j], util_L1[t,j],
                                     econ[t], econ_L1[t]) ;
      
      dev <- increment_deviance(dev, party[t,j], mu_post, sigma) ;
      party_rep[t,j] <- normal_rng(mu_post, sigma) ;      
      log_lik_mat[t,j] <- normal_log(party[t,j], mu_post, sigma) ;
    }
  }
  
  log_lik <- to_vector(log_lik_mat) ;
  } #end local

  resids_rep <- party - party_rep ; 

}



