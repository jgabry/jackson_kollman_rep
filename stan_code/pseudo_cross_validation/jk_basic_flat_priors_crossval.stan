/* From Ben Goodrich: sort of cross validation: 
A more general approach might be for one observation (at a time) to be considered missing, as opposed to left out. Then estimate the missing observation, along with the other unknowns, compare the posterior distribution of the missing observation to the actual data point, do that for all observations that you have, and average.
*/

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

  int<lower=1,upper=T> t_miss ; # time index for obs considered "missing"
  int<lower=1,upper=J> j_miss ; # group index for obs considered "missing"
}

transformed data {
  real            beta_mean ; # prior mean for betas
  real<lower=0>   beta_sd ; # prior sd for betas
  
  beta_mean <- 0.0 ;
  beta_sd <- 10.0 ;
}

parameters {  
  real<lower=0,upper=1>     rho[T] ; # time-varying autoregressive parameter
  real<lower=0,upper=10>    sigma ; # standard deviation
  vector[3]                 beta_noise ; 
  real                      D_0 ; # intercept
  real                      delta ; # coeff on error correction term
}

transformed parameters {
  vector[3] beta ;
  beta <- beta_mean + beta_sd*beta_noise ; 
}

model {
  # Priors
  
  beta_noise ~ normal(0,1) ; # implies beta[i] ~ normal(beta_mean, beta_sd)
  
  # D_0 ~ Unif(-Inf,Inf)
  # delta ~ Unif(-Inf,Inf)
  # sigma ~ Unif(0, 10)
  # rho[t] ~ Unif(0,1)
  
  
  # Likelihood
  
  for (j in 1:J) {
    for(t in 1:T) {
      if (j != j_miss && t != t_miss) {
      real mu ; 
      
      if (t == 1) 
        mu <- mean_wo_correction(D_0, rho[t], beta, party_L1[t,j], util[t,j], econ[t]) ;
      else 
        mu <- mean_w_correction(delta, D_0, rho[t], rho[t-1], beta,
                                party_L1[t,j], party_L2[t,j], 
                                util[t,j], util_L1[t,j],
                                econ[t], econ_L1[t]) ;
      
      increment_log_prob(normal_log(party[t,j], mu, sigma)) ;    
    }
    }
  }  
}

generated quantities {
  real party_miss_pred ; # prediction for "missing obs"

  { # local
  real mu_miss ;  
  if (t_miss == 1) 
    mu_miss <- mean_wo_correction(D_0, rho[t_miss], beta, 
                                  party_L1[t_miss,j_miss], 
                                  util[t_miss,j_miss], 
                                  econ[t_miss]) ;
  else 
    mu_miss <- mean_w_correction(delta, D_0, rho[t_miss], rho[t_miss-1], beta,
                                 party_L1[t_miss,j_miss], party_L2[t_miss,j_miss], 
                                 util[t_miss,j_miss], util_L1[t_miss,j_miss],
                                 econ[t_miss], econ_L1[t_miss]) ;

  party_miss_pred <- normal_rng(mu_miss, sigma) ;
  } # end local

}
