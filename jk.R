library(foreign)
library(plyr)
library(rstan)
# library(SHINYstan)

data <- read.dta("jk_processed_data.dta")
pid <- data$pid
pid1 <- data$pid1
pid2 <- data$pid2
util1 <- data$util1
util11 <- data$util11
t <- data$year
T <- length(unique(t)) # number of time periods

# make indicators for groups (south/black combinations)
Nwh <- with(data, south == 0 & black == 0)
Nbl <- with(data, south == 0 & black == 1)
Swh <- with(data, south == 1 & black == 0)
Sbl <- with(data, south == 1 & black == 1)
group_names <- c("Nwh", "Nbl", "Swh", "Sbl")
J <- length(group_names) # number of groups


# make some empty TxJ matrices
mat_names <- c("party", "party_L1", "party_L2", "util", "util_L1")
for (name in mat_names) {
  assign(name, matrix(NA, nrow = T, ncol = J, dimnames = list(NULL, group_names)))
}

for (j in 1:J) {
  grp <- get(group_names[j])
  party[, j] <- pid[grp]
  party_L1[, j] <- pid1[grp]
  party_L2[, j] <- pid2[grp]
  util[, j] <- util1[grp]
  util_L1[, j] <- util11[grp]
}


by_year <- ddply(data, "year", summarise, 
           retro1 = mean(retro1),
           retro11 = mean(retro11)
           )

econ <- by_year$retro1
econ_L1 <- by_year$retro11

stan_data <- list(
  party = party,
  party_L1 = party_L1,
  party_L2 = party_L2,
  util = util,
  util_L1 = util_L1,
  econ = econ,
  econ_L1 = econ_L1,
  J = J, # number of groups
  T = T # number of time periods 
  )

rm(list = setdiff(ls(), "stan_data"))
