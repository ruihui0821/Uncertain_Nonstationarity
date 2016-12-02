import matplotlib.pyplot as plt

import numpy as np
import math  # This will import math module
from scipy.stats import lognorm, norm
# A lognormal continuous random variable. logpdf(x, s, loc=0, scale=1), s is shape parameter logsigma, loc is locale parameter logmu, scale parameter is exp(logmu)
# A normal continuous random variable. pdf(x, loc=0, scale=1), cdf(x, loc=0, scale=1)

mu0 = 100.0
cv0 = 0.96
inmu = 2.0
insigma = 0.5

QCMIN = 0.01
QCMAX = 1000.0
deltaq = 0.01 # integration interval
inteq = np.arange( QCMIN, QCMAX + deltaq, deltaq )


# Converting the mean and variance to the location parameter logmu and the scale parameter logsigma of a lognormal distribution
mu = mu0 + mu0 * 0 * inmu  # Mean annual average peak flow (m**3/s), the bigger the lower peak probability and longer tail
cv = cv0 + cv0 * 0 * insigma# Coefficient of variation
sd = mu*cv # Standard deviation of annual average peak flow
var = sd**2 # Variation of flow, the bigger the lower peak probability and longer tail
logmu = math.log( ( mu**2) / math.sqrt( var + mu**2 ) ) # location parameter of annual peak flow
logsigma = math.sqrt( math.log( var / ( mu**2 ) + 1 ) ) # scale parameter of annual peak flow
expmu = math.exp(logmu) 

print mu, var, sd, logmu, logsigma, expmu

# Converting the location parameter logmu and the scale parameter logsigma of a lognormal distribution to mean and variance
#logmu = 4.45142783611 # the bigger the lower peak probability and longer tail
#logsigma = 0.554513029376 # the bigger the lower peak probability and longer tail
#expmu = math.exp(logmu)
#mu = math.exp(logmu + logsigma**2/2)
#var = (math.exp(logsigma**2) - 1) * math.exp(2*logmu + logsigma**2)
#sd = math.sqrt(var)
#print mu, var, sd, logmu, logsigma

plt.plot(inteq, lognorm.cdf(inteq, logsigma, loc = logmu, scale = expmu), lw=8, ls = 'solid', color = 'blue', alpha=0.6, label='lognorm cdf')
#plt.axis([QCMIN, QCMAX, 0.0, 0.01])
#http://matplotlib.org/api/pyplot_api.html
#linewidth or lw	float value in points
#linestyle or ls	['solid' | 'dashed', 'dashdot', 'dotted' | (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':' | 'None' | ' ' | '']

# If log(x) is normally distributed with mean logmu and variance logsigma**2, 
# then x is log-normally distributed with shape parameter logsigma and scale parameter exp(logmu).
# logpdf(x, s, loc=0, scale=1), s is shape parameter logsigma, loc is locale parameter logmu, scale parameter of lognormal is exp(logmu)

#print norm.cdf(logmu, logmu, logsigma)
#plt.plot(inteq, norm.pdf(inteq, logmu, expmu), lw=8, ls = 'solid', color = 'blue', alpha=0.6, label='lognorm pdf')

mu = mu0 + mu0 * 0 * inmu  # Mean annual average peak flow (m**3/s), the bigger the lower peak probability and longer tail
cv = cv0 + cv0 * 1 * insigma # Coefficient of variation
sd = mu*cv # Standard deviation of annual average peak flow
var = sd**2 # Variation of flow, the bigger the lower peak probability and longer tail
logmu = math.log( ( mu**2) / math.sqrt( var + mu**2 ) ) # location parameter of annual peak flow
logsigma = math.sqrt( math.log( var / ( mu**2 ) + 1 ) ) # scale parameter of annual peak flow
expmu = math.exp(logmu) 

plt.plot(inteq, lognorm.cdf(inteq, logsigma, loc = logmu, scale = expmu), lw=8, ls = 'dashed', color = 'orange', alpha=0.6, label='lognorm cdf')
#plt.axis([QCMIN, QCMAX, 0.0, 0.01])


mu = mu0 + mu0 * 1 * inmu  # Mean annual average peak flow (m**3/s), the bigger the lower peak probability and longer tail
cv = cv0 + cv0 * 0 * insigma # Coefficient of variation
sd = mu*cv # Standard deviation of annual average peak flow
var = sd**2 # Variation of flow, the bigger the lower peak probability and longer tail
logmu = math.log( ( mu**2) / math.sqrt( var + mu**2 ) ) # location parameter of annual peak flow
logsigma = math.sqrt( math.log( var / ( mu**2 ) + 1 ) ) # scale parameter of annual peak flow
expmu = math.exp(logmu) 

plt.plot(inteq, lognorm.cdf(inteq, logsigma, loc = logmu, scale = expmu), lw=8, ls = 'solid', color = 'red', alpha=0.6, label='lognorm cdf')
#plt.axis([QCMIN, QCMAX, 0.0, 0.01])


mu = mu0 + mu0 * 1 * inmu  # Mean annual average peak flow (m**3/s), the bigger the lower peak probability and longer tail
cv = cv0 + cv0 * 1 * insigma # Coefficient of variation
sd = mu*cv # Standard deviation of annual average peak flow
var = sd**2 # Variation of flow, the bigger the lower peak probability and longer tail
logmu = math.log( ( mu**2) / math.sqrt( var + mu**2 ) ) # location parameter of annual peak flow
logsigma = math.sqrt( math.log( var / ( mu**2 ) + 1 ) ) # scale parameter of annual peak flow
expmu = math.exp(logmu) 

plt.plot(inteq, lognorm.cdf(inteq, logsigma, loc = logmu, scale = expmu), lw=8, ls = 'dashed', color = 'purple', alpha=0.6, label='lognorm cdf')
#plt.axis([QCMIN, QCMAX, 0.0, 0.01])


mu = mu0 + mu0 * 2 * inmu  # Mean annual average peak flow (m**3/s), the bigger the lower peak probability and longer tail
cv = cv0 + cv0 * 0 * insigma # Coefficient of variation
sd = mu*cv # Standard deviation of annual average peak flow
var = sd**2 # Variation of flow, the bigger the lower peak probability and longer tail
logmu = math.log( ( mu**2) / math.sqrt( var + mu**2 ) ) # location parameter of annual peak flow
logsigma = math.sqrt( math.log( var / ( mu**2 ) + 1 ) ) # scale parameter of annual peak flow
expmu = math.exp(logmu) 

plt.plot(inteq, lognorm.cdf(inteq, logsigma, loc = logmu, scale = expmu), lw=8, ls = 'solid', color = 'green', alpha=0.6, label='lognorm cdf')
#plt.axis([QCMIN, QCMAX, 0.0, 0.01])


mu = mu0 + mu0 * 2 * inmu  # Mean annual average peak flow (m**3/s), the bigger the lower peak probability and longer tail
cv = cv0 + cv0 * 1 * insigma # Coefficient of variation
sd = mu*cv # Standard deviation of annual average peak flow
var = sd**2 # Variation of flow, the bigger the lower peak probability and longer tail
logmu = math.log( ( mu**2) / math.sqrt( var + mu**2 ) ) # location parameter of annual peak flow
logsigma = math.sqrt( math.log( var / ( mu**2 ) + 1 ) ) # scale parameter of annual peak flow
expmu = math.exp(logmu) 

plt.plot(inteq, lognorm.cdf(inteq, logsigma, loc = logmu, scale = expmu), lw=8, ls = 'dashed', color = 'black', alpha=0.6, label='lognorm cdf')
#plt.axis([QCMIN, QCMAX, 0.0, 0.01])



