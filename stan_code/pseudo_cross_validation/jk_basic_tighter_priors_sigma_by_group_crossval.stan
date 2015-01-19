# basic model with one sigma per group

/* From Ben Goodrich: sort of cross validation: A more general approach might
be for one observation (at a time) to be considered missing, as opposed to
left out. Then estimate the missing observation, along with the other
unknowns, compare the posterior distribution of the missing observation to the
actual data point, do that for all observations that you have, and average. 
*/

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

    vector cauchy_trans_vec2(real loc, real scale, vector noise) {
    vector[num_elements(noise)] out ;
    for (j in 1:num_elements(out)) {
      out[j] <- loc + scale * tan(pi() * (Phi_approx(noise[j]) - 0.5)) ;
    }
    return out ;
  }

  real normal_trans(real loc, real scale, real noise) {
    return loc + scale * noise ;
  }

  vector normal_trans_vec2(real loc, real scale, vector noise) {
    return loc + scale*noise ;
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
  vector[T]                     econ_L1 ; # econ, 1 lag

  int<lower=1,upper=T> t_miss ; # time index for obs considered "missing"
  int<lower=1,upper=J> j_miss ; # group index for obs considered "missing"
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
  vector<lower=0>[J]           Noise2 ;
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
  
  beta <- normal_trans_vec2(0.0, beta_scale, segment(Noise, 1, 3)) ;
  delta <- normal_trans(0.0, delta_scale, Noise[4]) ;
  D_0 <- normal_trans(0.0, D_0_scale, Noise[5]) ;
  sigma_party <- cauchy_trans_vec2(0.0, sigma_party_scale, Noise2) ;
}

model {
  Noise ~ normal(0,1) ; # each Noise[i] ~ N(0,1)
  Noise2 ~ normal(0,1) ; # each Noise2[i] ~ N(0,1) T[0,]
  # rho[t]  ~ unif(0,1)
  
  
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
      
      increment_log_prob(normal_log(party[t,j], mu, sigma_party[j])) ;    
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

  party_miss_pred <- normal_rng(mu_miss, sigma_party[j_miss]) ;
  } # end local
}




