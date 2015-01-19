# same as rho_logit_mvn, but with one sigma_party per group

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
  
  # transform noise ~ N(0,1) to cauchy 
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
  
  # inverse logit of vector elements
  vector inv_logit_vec(vector x) {
    vector[num_elements(x)] out ;
    for (j in 1:num_elements(out)) out[j] <- inv_logit(x[j]) ;
    return out ;
  }
  
}

data {
  int<lower=1>                  T ;         # number of time periods
  int<lower=1>                  J ;         # number of groups
  matrix[T,J]                   party ;     # mean partyid 
  matrix[T,J]                   party_L1 ;  # party, 1 lag
  matrix[T,J]                   party_L2 ;  # party, 2 lags
  matrix[T,J]                   util ;      # utility
  matrix[T,J]                   util_L1 ;   # # utility, 1 lag
  vector[T]                     econ ;      # retrospective economic evaluation
  vector[T]                     econ_L1 ;   # econ, 1 lag

  int<lower=1,upper=T> t_miss ; # time index for obs considered "missing"
  int<lower=1,upper=J> j_miss ; # group index for obs considered "missing"
}

transformed data {
  real<lower=0>                 beta_sd ;    
  real<lower=0>                 delta_sd ; 
  vector<lower=0>[J]            sigma_scale ;
  vector<lower=0>[T]            tau_scale ;
  
  beta_sd     <- 5.0 ;
  delta_sd    <- 2.5 ;
  sigma_scale <- rep_vector(2.5, J) ;
  tau_scale   <- rep_vector(2.5, T) ;
}

parameters {  
  real                          D_0 ;           # intercept
  cholesky_factor_corr[T]       L_Omega ;       # Choleski factor of correlation matrix
  vector[T]                     z_Omega ;       # N(0,1) 
  vector<lower=0>[T]            tau_noise ;     # N(0,1) 
  vector<lower=0>[J]            sigma_noise ;   # N(0,1) 
  vector[3]                     beta_noise ;    # N(0,1) 
  real                          delta_noise ;   # N(0,1) 
}

transformed parameters {
  vector<lower=0,upper=1>[T]    rho ;   # time-varying autoregressive parameter
  vector[3]                     beta ;  # regression coeffs
  real                          delta ; # coeff on error correction term
  vector<lower=0>[T]            tau ;   # scale for rho_logit 
  vector<lower=0>[J]            sigma ; # sd for party
  
  beta  <- beta_sd*beta_noise ; 
  delta <- delta_sd*delta_noise ;
  sigma <- cauchy_trans_vec(rep_vector(0, J), sigma_scale, sigma_noise) ;
  tau   <- cauchy_trans_vec(rep_vector(0, T), tau_scale, tau_noise) ;
  rho   <- inv_logit_vec(tau .* (L_Omega * z_Omega)) ;
}

model {
  ## Priors ##
  D_0 ~ normal(0, 1) ;
  delta_noise ~ normal(0, 1) ; # --> delta ~ normal(0, delta_sd)
  beta_noise ~ normal(0, 1) ;  # --> beta[i] ~ normal(0, beta_sd)
  sigma_noise ~ normal(0, 1) ; # --> sigma[i] ~ half-cauchy(0, sigma_scale)
  tau_noise ~ normal(0, 1) ;   # --> tau[i] ~ half-cauchy(0, tau_scale)
  z_Omega ~ normal(0, 1) ; 
  L_Omega ~ lkj_corr_cholesky(2) ;
  
  
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
      
      increment_log_prob(normal_log(party[t,j], mu, sigma[j])) ;    
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

  party_miss_pred <- normal_rng(mu_miss, sigma[j_miss]) ;
  } # end local
}
