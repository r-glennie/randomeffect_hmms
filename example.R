## Simple step-turn random walk HMM with random effects
## This example has an individual random effect on mean step length. 
## It can be extended to include random effects on any quantity (turning angle, transition
## probability), if the model can be supported by the data. 

# The model is written in TMB and must be compiled. Once compiled, you don't 
# need to recompile unless you change the TMB code. 
library(TMB)
compile("randomeffect_hmm.cpp", flags = "-Wno-ignored-attributes")
dyn.load(dynlib("randomeffect_hmm")) 

# SIMULATE DATA -----------------------------------------------------------

# set seed to reproduce simulation
set.seed(25832)

library(CircStats) # used to simulate wrapped cauchy turning angles 

n_states <- 2  # number of states 

## parameters true values 
step_mu <- c(10, 50) # mean step length in each state
step_sd <- c(5, 30) # standard deviation of step length 
rho <- c(0.2, 0.7) # concentration of turning angles 

## transition probability matrix for states
# the tpm is parameterised in this way in the numerical optimiser
ltpm <- qlogis(c(0.2, 0.3))
tpm <- diag(2)
tpm[!diag(2)] <- exp(ltpm)
tpm <- tpm / rowSums(tpm)
# compute equilibrum distribution of tpm 
# equilibrum distribution is left eigenvector of tpm 
stat_dist <- eigen(t(tpm))$vectors[,1]
stat_dist <- stat_dist / sum(stat_dist)

## Random effects: this is the variation in mean step on the log scale 
# So, population mean of 10 with step_ranef = 0.1, means roughly 95% 
# of individual mean step lengths will be between 10*exp(-2*0.1) = 8
# and 10*exp(2*0.1) = 12 
step_ranef <- 0.5

## T = number of regular time observations per individual
## n = number of individuals 
T <- 100
n <- 20 

## Simulate steps and turns 
# mean step length for each individual 
step_mu_r <- rnorm(n, 0, step_ranef)
step <- turn <- matrix(0, nr = T, nc = n)
id <- rep(1:n, each = T)

for (i in 1:n) {
  step_mu_i <- exp(log(step_mu) + step_mu_r[i])
  # convert step mean/sd to shape/scale of Gamma distribution 
  step_scale <- step_sd^2 / step_mu_i
  step_shape <- step_mu_i^2 / step_sd^2
  # initial state 
  s <- sample(1:2, prob = stat_dist, size = 1, replace = TRUE)
  for (t in 1:T) {
    # step and turn given state 
    step[t,i] <- rgamma(1, scale = step_scale[s], shape = step_shape[s])
    turn[t,i] <- rwrpcauchy(1, location = 0, rho = rho[s])
    # update state 
    s <- sample(1:2, prob = tpm[s,], size = 1)
  }
}
# turning angles between -pi and pi
turn[turn > pi] <- turn[turn > pi] - 2*pi

# vectorise 
step <- as.vector(step)
turn <- as.vector(turn)



# FIT MODEL WITH NO RANDOM EFFECTS  ---------------------------------------

# The model with random effects will be easier to fit if you have good 
# starting values, so fit a non-random effect model first. 

# number of states to fit to data 
fit_nstates <- 2 

# data list passed to TMB 
tmb_dat <- list(step = step, 
                      turn = turn, 
                      id = id, 
                      n_states = fit_nstates)

# parameters to pass to TMB, must match this order
# Notice, values are specified for random effect parameters but these 
# are ignored later. 
tmb_par <- list(log_step_mu = rep(log(mean(step)), fit_nstates), 
                log_step_sd = rep(log(sd(step)), fit_nstates), 
                logit_turn_rho = rep(0, fit_nstates), 
                ltpm = rep(0, fit_nstates^2 - fit_nstates), 
                step_r = rep(0, n), 
                log_step_ranef = 0) 

# Create model object and specify using map that we want to fix the random effects 
# to be their initial values (which are zero - so no random effects). This is specified
# by setting these valueus to factor NA in map. 
obj <- MakeADFun(data = tmb_dat, 
                 parameters = tmb_par, 
                 map = list(step_r = as.factor(rep(NA, n)), 
                            log_step_ranef = as.factor(NA)))

# fit model 
fit <- nlminb(start = obj$par, objective = obj$fn, gradient = obj$gr)

# compute variance components for parameter estimators 
res <- sdreport(obj)

# format estimates 
ests <- data.frame(est = res$value, sd = res$sd)
ests[1:4,] <- exp(ests[1:4,])
ests <- ests[-nrow(ests),]
ests <- round(ests, 2)
rownames(ests) <- c("stepmu1", "stepmu2", "stepsd1", "stepsd2", "rho1", "rho2", "tpm1-1", "tpm2-1", "tpm1-2", "tpm2-2")



# FIT WITH RANDOM EFFECTS  ------------------------------------------------

# set starting values based on non-ranef model 
tmb_par2 <- list(log_step_mu = res$par.fixed[1:2], 
                log_step_sd = res$par.fixed[3:4], 
                logit_turn_rho = res$par.fixed[5:6], 
                ltpm = res$par.fixed[7:8], 
                step_r = rep(0, n), 
                log_step_ranef = 0) 

# create ranef object, specify that step_r is the individual random effects. This 
# tells TMB to integrate them out using Laplace approximation. 
rf_obj <- MakeADFun(data = tmb_dat, 
                 parameters = tmb_par2, 
                 random = "step_r")

# fit model
rf_fit <- nlminb(start = rf_obj$par, objective = rf_obj$fn, gradient = rf_obj$gr)
# get variance components and random effect best predictions 
rf_res <- sdreport(rf_obj)

# format fixed effect estimates 
rf_ests <- data.frame(est = rf_res$value, sd = rf_res$sd)
rf_ests[1:4,] <- exp(rf_ests[1:4,])
rf_ests <- round(rf_ests, 2)
rownames(rf_ests) <- c("stepmu1", "stepmu2", "stepsd1", "stepsd2", "rho1", "rho2", "tpm1-1", "tpm2-1", "tpm1-2", "tpm2-2", "log_step_ranef")

# get predicted random effects 
pred_ranef <- rf_res$par.random
# plot true against predicted random effects 
plot(step_mu_r, pred_ranef)
abline(0, 1)







