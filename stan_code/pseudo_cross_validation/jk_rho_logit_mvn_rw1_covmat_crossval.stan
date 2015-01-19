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
  matrix[T,T]                   A ; # adjacency matrix
  matrix[T,T]                   D ; # degree matrix

  int<lower=1,upper=T> t_miss ; # time index for obs considered "missing"
  int<lower=1,upper=J> j_miss ; # group index for obs considered "missing"
}

transformed data {
  matrix[T,T] K ;
  K <- D - 0.99 * A ;
}

parameters {  
  vector[T]                 rho_logit ;
  real<lower=0>             tau_noise ; 
  real<lower=0>             sigma_noise ; 
  vector[3]                 beta_noise ; 
  real                      D_0 ; # intercept
  real                      delta_noise ; # coeff on error correction term
}

transformed parameters {
  real<lower=0>   tau ;
  vector[T]       rho ;
  vector[3]       beta ;
  real            delta ;
  real<lower=0>   sigma ;
  
  delta   <- 2.5 * delta_noise ;
  beta    <- 5.0 * beta_noise ; 
  sigma   <- cauchy_trans(0, 2.5, sigma_noise) ;
  tau     <- cauchy_trans(0, 1.0, tau_noise) ;

  rho <- inv_logit_vec(rho_logit) ;
}

model {
  
  # Priors
  beta_noise ~ normal(0, 1) ; # implies beta[i] ~ normal(beta_mean, beta_sd)
  D_0 ~ normal(0, 1) ;
  delta_noise ~ normal(0, 1) ;
  sigma_noise ~ normal(0, 1) ;
  tau_noise ~ normal(0, 1) ;

  
  rho_logit ~ multi_normal_prec(rep_vector(0,T), K/tau) ;
  
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
