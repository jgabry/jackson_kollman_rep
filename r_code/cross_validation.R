source("jk.R")
T <- stan_data$T
J <- stan_data$J

fit_compile <- stan(file = "stan/stan_code/cross_validation.stan")


nChain <- 4
nIter <- 2000
nWarmup <- 1500
stan_save <- matrix(NA, # 'empty' matrix to store results
                    nrow = nChain*(nIter - nWarmup), # number of saved samples
                    ncol = T*J) # number of observations 

save_column <- 1
for (t in 1:T) {
  for (j in 1:J) {
    stan_data$t_miss <- t
    stan_data$j_miss <- j
    fit <- stan(fit = fit_compile, 
                data = stan_data, 
                pars = "party_miss_pred",
                chains = nChain, iter = nIter, warmup = nWarmup) 
    
    stan_save[, save_column] <- extract(fit)$party_miss_pred
    
    print(paste0("Completed ", save_column, "/",T*J))
    save_column <- save_column + 1
    
  }
}

y_mat <- stan_data$party
y_vec <- y[1,]
for (t in 2:T) y_vec <- c(y_vec, y_mat[t,])

err_sq_mat <- stan_save
for (i in 1:ncol(err_sq_mat)) {
  err_sq_mat[,i] <- (stan_save[, i] - y_vec[i])^2
}

MSE <- apply(err_sq_mat, 2, mean)
RMSE <- sqrt(MSE)
meanRMSE <- mean(RMSE)




