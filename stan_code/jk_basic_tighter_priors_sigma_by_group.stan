# basic model with one sigma per group

functions {
  # mean without error correction term
  real mean_wo_correction(real D_0, real rho, vector beta, 
                          real party_L1, real util, real econ) {
    real val ;
    val <- D_0 + rho * party_L1 + (1 - rho) * (beta[1] + beta[2] * util + beta[3] * econ) ;
    return val ;
  }
  
  # error correction term
  real error_correction(real D_0, real rho_t1, vector beta, 
                        real party_L1, real party_L2, 
                        real util_L1, real econ_L1) {
    real val ; 
    val <- party_L1 - D_0 - rho_t1 * party_L2 - (1-rho_t1) * (beta[1] + beta[2]*util_L1 + beta[3]*econ_L1) ;
    return val ;
  }
  
  # mean with error correction term
  real mean_w_correction(real delta, real D_0, real rho, real rho_t1, vector beta,
                         real party_L1, real party_L2, real util, real util_L1,
                         real econ, real econ_L1) {
    
    real val ;
    real m ; 
    real V ;
    
    m <- mean_wo_correction(D_0, rho, beta, party_L1, util, econ) ;
    V <- error_correction(D_0, rho_t1, beta, party_L1, party_L2, util_L1, econ_L1) ;
    val <- m - delta * V ;
    return val ;
  }
  
  # increment deviance
  real increment_deviance(real dev, real y, real mu, real sigma) {
    real new_term ;
    new_term <- (-2.0) * normal_log(y, mu, sigma) ;
    return dev + new_term ; 
  }
}

data {
  int<lower=1>                  T ; # number of time periods
  int<lower=1>                  J ; # number of groups
  matrix[T,J]                   party ; # mean partyid 
  matrix[T,J]                   party_L1 ; # party, 1 lag
  matrix[T,J]                   party_L2 ; # party, 2 lags
  matrix[T,J]                   util ; # utility
  matrix[T,J]                   util_L1 ; # utility, 1 lag
  vector[T]                     econ ; # retrospective economic evaluation
  vector[T]                     econ_L1 ; # E, 1 lag
}

transformed data {
  # prior scales for some parameters 
  real<lower=0>                 beta_scale ; 
  real<lower=0>                 delta_scale ;
  real<lower=0>                 D_0_scale ;
  real<lower=0>                 sigma_party_scale ;
  
  beta_scale          <- 5.0 ;
  delta_scale         <- 2.5 ;
  D_0_scale           <- 1.0 ;
  sigma_party_scale   <- 2.5 ;
}

parameters {   
  real<lower=0,upper=1>        rho[T] ;      # time-varying autoregressive parameter
  vector[5]                    Noise ; 
  vector[J]                    Noise2 ;
}

transformed parameters {
  vector[3]                     beta ;        # regression coefficients
  real                          delta ;       # coefficient on error correction term
  real                          D_0 ;         # intercept 
  vector<lower=0>[J]            sigma_party ; # sd(party[,j])  
  
  
  # beta[i] ~ normal(0, beta_scale) 
  # delta ~ normal(0, delta_scale)
  # D_0 ~ normal(0, D_0_scale) 
  # sigma_party ~ half-cauchy(0, sigma_party_scale) 
  
  beta <- beta_scale * segment(Noise, 1, 3) ;
  delta <- delta_scale * Noise[4] ;
  D_0 <- D_0_scale * Noise[5] ;
  for (j in 1:J) sigma_party[j] <- sigma_party_scale * tan(pi() * (Phi_approx(Noise2[j]) - 0.5)) ;
}

model {
  Noise ~ normal(0,1) ; # each Noise[i] ~ N(0,1)
  Noise2 ~ normal(0,1) ; # each Noise2[i] ~ N(0,1) T[0,]
  # rho[t]  ~ unif(0,1)
  
  
  for (j in 1:J) {
    vector[T] mu ;
    mu[1] <- mean_wo_correction(D_0, rho[1], beta, party_L1[1,j], util[1,j], econ[1]) ;
    for(t in 2:T) {
      mu[t] <- mean_w_correction(delta, D_0, rho[t], rho[t-1], beta,
                                 party_L1[t,j], party_L2[t,j], 
                                 util[t,j], util_L1[t,j],
                                 econ[t], econ_L1[t]) ;
      
    }
    increment_log_prob(normal_log(col(party,j), mu, sigma_party[j])) ;      
  }
  
}

generated quantities {
  real        dev ;           # deviance
  matrix[T,J] party_rep ;     # simulated values
  matrix[T,J] resids_rep ;    # residuals: party[t,j] - party_rep[t,j]
  vector[T*J] log_lik ;

  dev <- 0.0 ; 

{ # local
  matrix[T,J] log_lik_mat ;

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
      
      dev <- increment_deviance(dev, party[t,j], mu_post, sigma_party[j]) ;
      party_rep[t,j]    <- normal_rng(mu_post, sigma_party[j]) ; 
      log_lik_mat[t,j] <- normal_log(party[t,j], mu_post, sigma_party[j]) ;
    }
  }
  
  log_lik <- to_vector(log_lik_mat) ;
} #end local

  resids_rep <- party - party_rep ;
}




