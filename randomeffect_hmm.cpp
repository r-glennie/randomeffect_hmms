#include <TMB.hpp>

// Density for a wrapped Cauchy 
template<class Type> 
Type dwrpCauchy(Type a, Type rho) { 
  Type p = -log(rho); 
  Type pdf = sinh(p) / (Type(2.0 * M_PI) * (cosh(p) - cos(a))); 
  return(pdf); 
}

// Compute Negative-Log likelihood for random walk HMM with random effects
// DATA: 
//   step: observed step lengths 
//   turn: observed turning angles 
//   id: individual ID for each observation 
//   n_states: number of states 
// PARAMETERS: 
//   log_step_mu: log mean step length 
//   log_step_sd: log step length standard deviation
//   logit_turn_rho: turning angle concentration 
//   ltpm: non-diagonal inverse mlogit transition probabilities 
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Read in data 
  DATA_VECTOR(step);
  DATA_VECTOR(turn); 
  DATA_IVECTOR(id); 
  DATA_INTEGER(n_states);
  
  // Read in parameters
  PARAMETER_VECTOR(log_step_mu);
  PARAMETER_VECTOR(log_step_sd); 
  PARAMETER_VECTOR(logit_turn_rho); 
  PARAMETER_VECTOR(ltpm);
  PARAMETER_VECTOR(step_r); 
  PARAMETER(log_step_ranef); 
  
  vector<Type> turn_rho = Type(1.0) / (Type(1.0) + exp(-logit_turn_rho));  
  Type step_ranef = exp(log_step_ranef); 
  matrix<Type> tpm(n_states, n_states);
  int cur = 0;
  for (int i = 0; i < n_states; ++i) {
    tpm(i, i) = 1; 
   for (int j = 0; j < n_states; ++j) {
    if (i != j) {
      tpm(i, j) = exp(ltpm[cur]); 
      ++cur; 
    } 
   } 
   tpm.row(i) /= tpm.row(i).sum();
  } 
  // Compute stationary distribution
  matrix<Type> delta(1, n_states); 
  matrix<Type> I = matrix<Type>::Identity(n_states, n_states);
  matrix<Type> tpminv = I; 
  tpminv -= tpm; 
  tpminv = (tpminv.array() + 1).matrix(); 
  matrix<Type> ivec(1, n_states); for (int i = 0; i < n_states; ++i) ivec(0, i) = 1;
  // if tpm is ill-conditioned then just use uniform initial distribution 
  try {
    tpminv = tpminv.inverse();
    delta = ivec * tpminv;
  } catch(...) {
    for (int i = 0; i < n_states; ++i) delta(0, i) = 1.0 / n_states; 
  }
  // compute observation probabilities
  int n = step.rows();  
  matrix<Type> prob(n, n_states); 
  for (int s = 0; s < n_states; ++s) {
    Type step_sd = exp(log_step_sd(s)); 
    for (int i = 0; i < n; ++i) {
      Type step_mu = exp(log_step_mu(s) + step_r(id(i) - 1)); 
      prob(i, s) = dgamma(step(i), step_mu*step_mu/(step_sd*step_sd), (step_sd*step_sd/step_mu));
      prob(i, s) *= dwrpCauchy(turn(i), turn_rho(s)); 
    }
  } 
  // compute log-likelihood 
  Type llk = 0;
  matrix<Type> phi(delta);
  Type sumphi = 0;
  Type curid = id(0); 
  for (int i = 0; i < n; ++i) {
    if (curid != id(i)) phi = delta; 
    curid = id(i);
    phi = (phi.array() * prob.row(i).array()).matrix();  
    phi = phi * tpm;
    sumphi = phi.sum();
    llk += log(sumphi);
    phi /= sumphi;
  }
  // random effects
  for (int i = 0; i < step_r.size(); ++i) {
    llk += dnorm(step_r(i), Type(0.0), step_ranef, 1); 
  }
  
  Type nll = -llk; 
  
  // report 
  ADREPORT(log_step_mu); 
  ADREPORT(log_step_sd); 
  ADREPORT(turn_rho); 
  ADREPORT(tpm); 
  ADREPORT(step_ranef); 
  return nll;
}
