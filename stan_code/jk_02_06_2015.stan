# gaussian process prior on logit(rho)
# also estimate params of gp covariance function

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
  
  // matern 5/2 covariance function
  matrix cov_matern52(int T, real jitter) {
    matrix[T, T] X ;
    
    for (t1 in 1:T) {
      for (t2 in 1:t1) {
        real d ;
        real d_sqrt5 ; 
        d <- fabs(t1 - t2) ;
        d_sqrt5 <- sqrt(5) * d ;     
        X[t1,t2] <- (1 + d_sqrt5 + pow(d_sqrt5, 2) / 3) * exp(-d_sqrt5) ; 
        if (t1 != t2) 
          X[t2,t1] <- X[t1,t2] ;
        else 
          X[t1,t1] <- X[t1,t1] + jitter ; 
      }
    }
    
    return X ;
  }  
  
  matrix cov_gen_sq_exp(int T, vector pars) {
    matrix[T, T] X ;
    
    // off diagonal elements
    for (t1 in 1:(T-1)) {
      for (t2 in (t1+1):T) {
        X[t1,t2] <- pars[1] * exp(-pars[2] * pow(t1 - t2,2)) ;
        X[t2,t1] <- X[t1,t2] ;
      }
    }
    
    // diagonal elements
    for (t in 1:T) X[t,t] <- pars[1] + pars[3] ; 
    
    return X ;
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
#  int<lower=1,upper=2>          covariance_type ; # which gp covariance function to use                           
}

parameters {
  vector[T]                 rho_logit ;
  vector<lower=0>[3]        gp_pars_noise ;
  real<lower=0>             sigma_noise ; 
  vector[3]                 beta_noise ; 
  real                      D_0 ; # intercept
  real                      delta_noise ; # coeff on error correction term
}

transformed parameters {
  vector<lower=0>[3] gp_pars ;
  vector[T]       rho ;
  vector[3]       beta ;
  real            delta ;
  real<lower=0>   sigma ;
  
  delta   <- 2.5 * delta_noise ;
  beta    <- 5.0 * beta_noise ; 
  sigma   <- cauchy_trans(0, 2.5, sigma_noise) ;

  rho <- inv_logit_vec(rho_logit) ;
  for (j in 1:3) 
    gp_pars[j] <- cauchy_trans(0.0, 5.0, gp_pars_noise[j]) ;
}

model {

  matrix[T,T] SIGMA ; # covariance matrix
  
  SIGMA <- cov_gen_sq_exp(T, gp_pars) ;
  
  # Priors
  gp_pars_noise ~ normal(0,1) ;
  rho_logit ~ multi_normal(rep_vector(0,T), SIGMA) ;

  beta_noise ~ normal(0, 1) ; # --> beta[i] ~ normal(beta_mean, beta_sd)
  D_0 ~ normal(0, 1) ;
  delta_noise ~ normal(0, 1) ;
  sigma_noise ~ normal(0, 1) ;
  
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
  real        dev ; # deviance
  matrix[T,J] mu_posterior ; # posterior means (fitted values)
  matrix[T,J] party_rep ; # simulated values from posterior predictive dist.
  matrix[T,J] resids_rep ; # residuals: party[t,j] - party_rep[t,j]
  vector[T*J] log_lik ; # log likelihood (for computing waic)
  
{ # local 
  matrix[T,J] log_lik_mat ;
  
  dev <- 0.0 ;
  for (j in 1:J) {
    for(t in 1:T) {
      if (t == 1) 
        mu_posterior[t,j] <- mean_wo_correction(D_0, rho[t], beta, 
                                                party_L1[t,j], util[t,j], econ[t]) ;
      else 
        mu_posterior[t,j] <- mean_w_correction(delta, D_0, rho[t], rho[t-1], beta,
                                               party_L1[t,j], party_L2[t,j], 
                                               util[t,j], util_L1[t,j],
                                               econ[t], econ_L1[t]) ;
      
      dev <- increment_deviance(dev, party[t,j], mu_posterior[t,j], sigma) ;
      party_rep[t,j] <- normal_rng(mu_posterior[t,j], sigma) ;      
      log_lik_mat[t,j] <- normal_log(party[t,j], mu_posterior[t,j], sigma) ;
    }
  }
  
  log_lik <- to_vector(log_lik_mat) ;
} #end local

resids_rep <- party - party_rep ; 

}


