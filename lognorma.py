import matplotlib.pyplot as plt

import numpy as np
import math  # This will import math module
from scipy.stats import lognorm # A normal continuous random variable. pdf(x, loc=0, scale=1), cdf(x, loc=0, scale=1)

# Converting the mean and variance to the location parameter logmu and the scale parameter logsigma of a lognormal distribution
mu = 45.6042083208 # Mean annual average peak flow (m**3/s)
var = 1864.45056541 # Variation of flow
sd = math.sqrt(var) # Standard deviation of annual average peak flow
logmu = math.log( ( mu**2) / math.sqrt( var + mu**2 ) ) # location parameter of annual peak flow
logsigma = math.sqrt( math.log( var / ( mu**2 ) + 1 ) ) # scale parameter of annual peak flow 

print mu, var, sd, logmu, logsigma

# Converting the location parameter logmu and the scale parameter logsigma of a lognormal distribution to mean and variance
logmu = 3.5 # the bigger the lower peak probability and longer tail
logsigma = 0.8 # the bigger the lower peak probability and longer tail
expmu = math.exp(logmu)
mu = math.exp(logmu + logsigma**2/2)
var = (math.exp(logsigma**2) - 1) * math.exp(2*logmu + logsigma**2)
sd = math.sqrt(var)
print mu, var, sd, logmu, logsigma

QCMIN = 0.01
QCMAX = 200.0
deltaq = 0.01 # integration interval
inteq = np.arange( QCMIN, QCMAX + deltaq, deltaq )

plt.plot(inteq, lognorm.pdf(inteq, logsigma, loc = logmu, scale = expmu), lw=2, alpha=0.6, label='lognorm pdf')

# If log(x) is normally distributed with mean logmu and variance logsigma**2, 
# then x is log-normally distributed with shape parameter logsigma and scale parameter exp(logmu).
# logpdf(x, s, loc=0, scale=1), s is shape parameter logsigma, loc is locale parameter logmu, scale parameter is exp(logmu)
