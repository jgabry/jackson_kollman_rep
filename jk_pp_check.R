jk_pp_check <- function(stanfit, y, make_dir = TRUE) { 
  # y is TxJ matrix of observed data (J is number of groups)
  # plot_dir is directory to save plots
  
  graphics.off()
  print("Generating plots ...")
  
  stanfit_name <- deparse(substitute(stanfit))
  plot_dir <- paste0("plots/", stanfit_name)
  
  if (make_dir == TRUE) {
    print(paste("Creating directory", plot_dir))
    dir.create(plot_dir)
  }
  
  plot_dir <- paste0(plot_dir,"/")
  
  pdf(file = paste0(plot_dir, "ppcheck_plots.pdf"))
  
  
  group_names <- c("Nwh", "Nbl", "Swh", "Sbl")
  
  # extract replications and residuals
  y_rep <- extract(stanfit, pars = "party_rep")[[1]]
  resids_rep <- extract(stanfit, pars = "resids_rep")[[1]]  
  
  # compute averages for later use
  avg_rep <- apply(y_rep, c(2,3), mean)
  avg_resid_rep <- apply(resids_rep, c(2,3), mean)
  
  
  
  #### Density: residuals ####
  plot(density(resids_rep), lwd = 2,
       axes = FALSE, ylab = "", xlab = "",
       main = "Kernel density estimate of residuals")
  axis(1, lwd = 4)
  
  #### Scatter: y vs avg_rep ####
  plot(y, avg_rep, pch = 20, xlab = "Observed", ylab = "Avg. simulated", col = "purple", axes = FALSE)
  abline(a = 0, b = 1, lwd = 2, lty = 2, col = "gray35")
  axis(1, lwd = 4)
  axis(2, lwd = 4)
  
  #### Density: subset of simulated vs observed ####
  samp_size <- 20
  y_rep_samp <- y_rep[sample(nrow(y_rep), size = samp_size) ,,]
  ymax <- max(c(density(y_rep_samp)$y, density(y)$y))
  
  plot(density(y_rep_samp[1,,]), ylim = c(0, ymax), 
       axes = FALSE, xlab = "", ylab = "", 
       main = "Kernel density estimates (obs vs sample of reps)")
  for (i in 2:20) lines(density(y_rep_samp[i,,]), col = "gray35")
  lines(density(y), col = "purple", lwd = 3)
  axis(1, lwd = 4)
  legend("bottom", c("obs", "rep"), lwd = 2, col = c("purple", "black"), bty = "n")
  
  
  #### Scatter: avg_rep vs avg_resid_rep
  plot(avg_rep, avg_resid_rep, pch = 20, col = "purple",
       axes = FALSE, main = "Average replicated vs average residual",
       xlab = "Avg. simulated", ylab = "Avg. simulated residual")
  axis(1, lwd = 4)
  axis(2, lwd = 4)
  
  
  #### Histogram: replications ####
  hist(y_rep, freq = FALSE, 
       axes = FALSE, ylab = "", xlab = "",
       main = "Histogram of replications",
       sub = list("Observed values plotted in purple along line y = 0 \n Dashed lines are medians", cex = 0.7),
       border = "white", col = "skyblue")
  abline(v = median(y_rep), lty = 2, lwd = 1, col = "skyblue4")
  points(y, rep(0, times = length(y)), cex = 0.75, col = "purple")
  abline(v = median(y), lty = 2, col = "purple")
  axis(1, lwd = 4)
  
  
  
  ####  Histograms: test statistics #### 
  par(mfrow = c(2,2)) 
  funs <- c("mean", "sd", "min", "max")
  Tstats <- lapply(seq_along(funs), function(f) apply(y_rep, 1, funs[f]))
  names(Tstats) <- funs
  for (f in seq_along(funs)) {
    hist(Tstats[[f]], freq = FALSE, xlab = "", ylab = "", axes = FALSE, 
         border = "white", col = "skyblue", main = funs[f])
    abline(v = do.call(funs[f], args = list(y)), lwd = 3, col = "purple")
    axis(1, lwd = 4)
  }
  par(mfrow = c(1,1))
  mtext("Histograms of test statistics", side = 3, line = 2)
  
  
  #### Histograms: mean by group ####
  means_by_group <- apply(y_rep, c(1,3), mean)
  colnames(means_by_group) <- group_names
  par(mfrow = c(2,2))
  for (i in 1:ncol(means_by_group)) {
    hist(means_by_group[,i], freq = FALSE, 
         axes = FALSE, main = "",
         xlab = paste("mean partisanship: ", group_names[i]), ylab = "",
         border = "white", col = "skyblue")
    abline(v = mean(y[,i]), lwd = 2, col = "purple")
    axis(1, lwd = 4)
  }
  par(mfrow = c(1,1))
  mtext("Mean partisanship by group", side = 3, line = 2)
  
  
  #### Histograms: mean by time period ####
  means_by_time <- apply(y_rep, c(1,2), mean)
  par(mfrow = c(5,4))
  for (i in 1:ncol(means_by_time)) {
    hist(means_by_time[,i], freq = FALSE, 
         axes = FALSE, main = "",
         xlab = paste("mean partisanship: t =",i), ylab = "",
         border = "white", col = "skyblue")
    abline(v = mean(y[i,]), lwd = 2, col = "purple")
    axis(1, lwd = 4)
  }
  par(mfrow = c(1,1))
  mtext("Mean partisanship by time period", side = 3, line = 2)
  
  
  #### Time-series: data vs reps over time ####
  
  # obs vs 20 replications
  par(mfrow = c(2,1))
  sim_ids <- sample(1:nrow(y_rep), 20)
  plot(ts(rowMeans(y_rep[sim_ids[1],,])), axes = FALSE,
       lwd = 0.75, ylim = c(0, 2),
       main = "Observed data (purple) and \n 20 posterior predictive simulations (gray)", xlab = "t", ylab = "")
  for (sim in sim_ids[-1]) {
    lines(ts(rowMeans(y_rep[sim,,])), lwd = 0.75) 
  }
  lines(ts(rowMeans(y)), col = "purple", lwd = 2)
  axis(1, lwd = 4)
  axis(2, lwd = 4)
  
  # plot obs (in purple) vs all sims
  plot(ts(rowMeans(y_rep[1,,])), axes = FALSE,
       lwd = 0.1, col = "lightgray", ylim = c(0, 2),
       main = "Observed data (purple) and \n all posterior predictive simulations (gray)", xlab = "t", ylab = "")
  for (i in 2:nrow(y_rep)) {
    lines(ts(rowMeans(y_rep[i,,])), lwd = 0.1, col = "lightgray") 
  }
  lines(ts(rowMeans(y)), col = "purple", lwd = 2)
  axis(1, lwd = 4)
  axis(2, lwd = 4)
  par(mfrow = c(1,1))
  
  
  
  #### Time-series: data vs reps over time by group ####
  par(mfrow = c(2,2))
  
  for (j in 1:4) {
    yrange <- c(min(y_rep[,,j], y[,j]), max(y_rep[,,j], y[,j]))
    # plot obs (in purple) vs all sims
    plot(ts(y_rep[1,,j]), axes = FALSE,
         lwd = 0.1, col = "lightgray", ylim = yrange,
         main = "",
         sub = paste("Group:", group_names[j]),
         xlab = "t", ylab = "")
    for (i in 2:nrow(y_rep)) {
      lines(ts(y_rep[i,,j]), lwd = 0.1, col = "lightgray") 
    }
    lines(ts(y[,j]), col = "purple", lwd = 2)
    axis(1, lwd = 4)
    axis(2, lwd = 4)
  }
  par(mfrow = c(1,1))
  mtext("Observed data (purple) and \n all posterior predictive simulations (gray)", 
        side = 3, line = 2)
  
  
  
  graphics.off()
  
  
  print(paste("Plots saved to", plot_dir))
  
}