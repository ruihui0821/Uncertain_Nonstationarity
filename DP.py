import time
start = time.time()

import numpy as np
import math  # This will import math module
from scipy.stats import lognorm # A lognormal continuous random variable. lognorm.pdf(x, s) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)
# Specifically, lognorm.pdf(x, s, loc, scale) is identically equivalent to lognorm.pdf(y, s) / scale with y = (x - loc) / scale.
from scipy.integrate import quad, simps
import csv
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

largenumber = 1e+20 # a very large number for calculation purpose

# Planning time periods
T = 20 # the year interval that a levee to be upgrading (yrs)
NT = 5 # the number of intervals in a levee's lifetime
n = int(NT * T) # a levee total lifetime (yrs)

# Levee design standard
H0 = 0.0 # The initial levee height (meter)
DELTAH = 0.1 # Step size of levee height (meter)
Hmin = 0.0 # The minimum levee height (meter)
Hmax = 10.0 # The maximum levee height (meter)
Hupmax = 10.0 # The maximum upgrading levee height (meter)

# Financial Parameters
DC = 8.0 # Flood Damage Cost, assuming simultaneous failures on two riversides with a constant value (million $)
Cland = 1.0 # Price of land ($/m**2) 
Csoil = 30.0 # Soil compaction cost or construction cost per unit levee material ($/m**3)
Cadjust =1.3 # Soft Cost Multiplier considering management
R = 0.04 # Real (inflation-adjusted) discount rate

# Stationary Annual Flow Parameters
mu0 = 100.0 # Mean annual flow (m**3/s), the bigger the lower peak probability and longer tail
sigma0 = 66.0 # Standard deviation of annual flow
cv0 = sigma0 / mu0 # Coefficient of variation
var0 = sigma0**2 # Variation of flow, the bigger the lower peak probability and longer tail
logmu0 = math.log( ( mu0**2) / math.sqrt( var0 + mu0**2 ) ) 
# location parameter of annual flow # the bigger the lower peak probability and longer tail
logsigma0 = math.sqrt( math.log( var0 / ( mu0**2 ) + 1 ) ) 
# scale parameter of annual flow  # the bigger the lower peak probability and longer tail
expmu0 = math.exp(logmu0)  

deltaq = 0.01 # integration interval

# climate scenarios
NMU = 1 # number of different changing mean
NSIGMA = 1 # number of different changing standard deviation
NA = NMU * NSIGMA # number of climate scenarios

muslope = 0.0/100 
# initialization of variation of possible annual average peak flow   
sigmaslope = 0.0/100     

MU = np.zeros(n)
# initialization of mean annual flow for n years
SIGMA = np.zeros(n)
# initialization of standard deviation of annual flow for n years
LOGMU = np.zeros(n)
# initialization of location parameter of annual peak flow for n years
LOGSIGMA = np.zeros(n)
# initialization of scale parameter of annual peak flow for n years
# On a logarithmic scale, {\displaystyle \mu } \mu  and {\displaystyle \sigma } \sigma  can be called the location parameter and the scale parameter, respectively.
for i in range(n):
# stage or time i        
    MU[i] = mu0 + muslope * mu0 * (i + 1)    
    # annual average peak flow for time i, changing linearly
    SIGMA[i] = sigma0 + sigmaslope * sigma0 * (i+1)
    # standard deviation of annual flow for time i, changing linearly
    cv = SIGMA[i] / MU[i]
    # additional change to Coefficient of variation, not necessary since sigma will change with mu anyway      
    var = SIGMA[i]**2 
    # Variation of flow, the bigger the lower peak probability and longer tail
    LOGMU[i] = math.log( ( (MU[i])**2) / math.sqrt( var + (MU[i])**2 ) )         
    # location parameter of annual flow for time i and climate scenario j
    LOGSIGMA[i] = math.sqrt( math.log( var / ( (MU[i])**2 ) + 1 ) )
    # scale parameter of annual flow for time i and climate scenario j
    #print i+1, k+1, MU[i,k], SIGMA[i,k], cv, var, LOGMU[i,k], LOGSIGMA[i,k]    

# Given Channel Parameters
Sc = 0.0005 # Longitudinal slope of the Channel, also the slope of the hydraulic grade line or the linear hydraulic head loss
Nc = 0.05 # Roughness factor of the channel section, also known as the Manning coefficient
kconversion = 1.0 # A conversion factor k = 1.4859 for English Unit

# Levee geometry parameters
W = 90.0 # Total width till the toe of the levee (meter)
B = 60.0 # Total Channel width (meter)
D = 1.0 # Channel Depth (meter)
WS = 1.0/2.0 # Water side-slopes tan(waterside angle)= 1:2
LS = 1.0/4.0 # Land side-slopes tan(langdside angle)= 1:4
FS = 0.01 # Floodplain slope
Bc = 10.0 # Levee crown width (meter)
L = 2000.0 # Total levee length (meter)
 
# Levee Design parameters
NH = int(round((Hmax-Hmin)/DELTAH+1)) # Number of discretized initial levee height
NHUP = int(round((Hupmax)/DELTAH+1)) # Number of discretized upgrading levee height

EXH = np.zeros((n,NH))
# initialization of the discretized existing levee height as state
# (EXH[i][l]) i =1:n stage or time; l =1:NH discretized existing levee height
for i in range(n):
# stage or time i      
    for l in range(NH):
    # index for existing levee height l
        (EXH[i][l]) = DELTAH * l # Assign all possible existing heights

UPH = np.zeros((n,NHUP))
# initialization of the discretized upgrading levee height
# (UPH[i][m]) i =1:n stage or time;  m =1:NHUP discretized upgrading levee height
for i in range(n):
# stage or time i      
    for m in range(NHUP):
    # index for upgrading levee height m
        if ( i % T ) == 0:
        # annual expected total cost at upgrading time period every T year interl, including expected damage cost and construction cost
            (UPH[i][m]) = DELTAH * m # Assign all possible upgrading heights
        else:
        # annual expected total cost at none upgrading time periods, upgradh = 0, only expected damage cost
            (UPH[i][m]) = 0 # Assign all possible upgrading heights

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cost function including annualized construction cost and expected annual damage cost
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Basic calcuation for leveed channel
Hfp = ( W - B ) * FS # Floodplain height
Htoe = D + Hfp # Water level at the toe of the levee

# Flow corresponding to aNY specific water level (from bottom of the river), calculated by Manning's Equation
def flow(waterh):
    if waterh >= Htoe: # If water level is above the toe and below the top of the levee
        CrossSection = B * D + ( B + W ) * Hfp/2 + ( W + W + 2*( waterh - Htoe )/WS ) * ( waterh - Htoe )/2
        # Cross section area of flow at waterh depth
        WettedP = B + 2 * D + 2 * math.sqrt(( ( W - B )/2)**2 + (Hfp)**2) + 2 * math.sqrt(( ( waterh - Htoe )/WS )**2 + ( waterh - Htoe )**2)
        # Wetted perimeter
    else:
        if waterh >= D: # If water level is above the channel depth and below the toe of the levee
            CrossSection = B * D + ( B + B + 2 * ( waterh - D )/FS ) * ( waterh - D )/2
            # Cross section area of flow at waterh depth
            WettedP = B + 2 * D + 2 * math.sqrt(( ( waterh - D )/FS )**2 + ( waterh - D )**2)
            # Wetted perimeter
        else:
            CrossSection = waterh * B # Cross section area of flow at waterh depth
            WettedP = 2 * waterh + B # Wetted perimeter
    Velocity = kconversion/Nc * (CrossSection/WettedP)**(2/3) * math.sqrt(Sc) # Water velocity
    overflow = Velocity * CrossSection # Flow
    return overflow

QCMIN = flow( Htoe ) # The base channel capacity at the toe of the levee, below which there's no risk
# QCMIN = Qfp  The same as above, flow within the top of the floodplain

##################  Non-overflow intermediate failure probability
def integrand(Q, intemu, intesigma, inteheight):
    Htop = Htoe + inteheight
    QCMAX = flow( Htop )
    inteexpmu = math.exp(intemu)
    PQ = lognorm.pdf(Q, intesigma, loc = intemu, scale = inteexpmu) # # Probability of Flow Q (PDF)
    if QCMAX == QCMIN:
        PNOVER = 1
    else:
        PNOVER = ( Q - QCMIN )/( QCMAX - QCMIN )
    return PQ*PNOVER
    
def intermediatefailure(intemu, intesigma, inteheight):
    Htop = Htoe + inteheight
    QCMAX = flow( Htop )
    #FINTERMEDIATE = quad(integrand, QCMIN, QCMAX, args=(intemu, intesigma, inteheight))[0]
    inteq = np.arange( QCMIN, QCMAX + deltaq, deltaq )
    FINTERMEDIATE = simps(integrand(inteq, intemu, intesigma, inteheight), inteq)
    return FINTERMEDIATE

 
##################  Overflow failure probability
def overtopfailure(overmu, oversigma, overheight):
    Htop = Htoe + overheight
    OVERFLOW = flow(Htop) # Overtopping Flow
    overexpmu = math.exp(overmu)
    FNOVER = lognorm.cdf(OVERFLOW, oversigma, loc = overmu, scale = overexpmu)
    FOVER = 1 - FNOVER    
    return FOVER

################  Annually construction or upgrading costs of a levee, based on existing height and upgrading height
def constructioncost(conexisth, conupgradh):
    ALEVEE0 = ( Bc + Bc + conexisth * (1/WS + 1/LS)) * conexisth/2
    # Cross section area of the existing levee (m**2)
    ALEVEE1 = ( Bc + Bc + (conupgradh + conexisth) * (1/WS + 1/LS)) * (conupgradh + conexisth)/2
    # Cross section area of the upgraded levee (m**2)
    ALEVEE = ALEVEE1 - ALEVEE0
    # Upgrading cross section area of one levee (m**2)
    V = ALEVEE * L #Total volume of one levee (m^3)
    LC0 = Cland * L * ( Bc + ( 1/WS + 1/LS ) * conexisth )
    # Land use cost of the existing levee primarily for purchasing land ($)
    LC1 = Cland * L * ( Bc + ( 1/WS + 1/LS ) * (conupgradh + conexisth) )
    # Land use cost of the upgraded levee primarily for purchasing land ($)
    LC = LC1 - LC0
    # Upgrading land use cost of one levee ($)
    cc = ( Cadjust * Csoil * V ) + LC # levee construction cost
    return cc
     
def annualtotalcost(EXISTH, UPGRADH, LOGMU, LOGSIGMA):
    cc = constructioncost(EXISTH, UPGRADH)
    CC = 2 * cc / 1000000.0 # construction cost of two levees in million $
    HEIGHT = EXISTH + UPGRADH
    FINTERMEDIATE = intermediatefailure(LOGMU, LOGSIGMA, HEIGHT)
    FOVER = overtopfailure(LOGMU, LOGSIGMA, HEIGHT)
    EAD = DC * (FINTERMEDIATE + FOVER) # expected annual damage cost in million $    
    # Expected annual total cost    
    EAT = CC + EAD
    return EAT


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SDP Program
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
                
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#STEP 2 Calculation of the backward DP value functions

EATC = np.ones((( n, NH, NHUP ))) * largenumber
# initialization of the expected annual total cost for upgrading height m=1:NHUP and existing height l=1:NH; at time i=1:n 
# (EATC[i][l][m]) i =1:n stage or time; l =1:NH existing levee height; m =1:NHUP upgrading levee height
VALUE = np.ones((( n, NH, NHUP ))) * largenumber # a very large number
# initialization of the accumulation for the decisions from all later stages, 
# for upgrading height m=1:NHUP and existing height l=1:NH, at time i=1:n
# (VALUE[i][l][m]) i =1:n stage or time; l =1:NH existing levee height; m =1:NHUP upgrading levee height
EMINIVALUE = np.ones((( n, NH, NHUP ))) * largenumber # a very large number
# initialization of the expection of the next stage minimum value for accumulation  
# at time i=1:n, with current existing height l=1:NH and upgrading height m=1:NHUP
# (EMINIVALUE[i][l][m]) i =1:n stage or time; l =1:NH existing levee height; m =1:NHUP upgrading levee height

# backward recursive function for the last stage
for i in range(n-1,n):
# stage or time i, backward index(n-1-i), starting from the last stage index(n-1)              
    for l in range(NH):
    # existing levee height l
        existh = (EXH[i][l])
        if ( i % T ) == 0:
            NHUPm = int(round((Hupmax)/DELTAH+1)) # upgrading time periods, calculate for all possible upgrading heights (UPH[i][m]) = 0
        else:
            NHUPm = 1 # none upgrading time periods, upgradh = 0, so could only calculate for the first (UPH[i][1]) = 0
        for m in range(NHUPm):
        # upgrading levee height m
            upgradh = (UPH[i][m])
            # corresponding to (EXH[i+1][]) = (EXH[i][l])+(UPH[i][m])
            logmu = (LOGMU[i])
            logsigma = (LOGSIGMA[i])                 
            if existh + upgradh <= Hmax + 1e-6:
            # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                (EATC[i][l][m]) = annualtotalcost( existh, upgradh, logmu, logsigma )
            (VALUE[i][l][m]) = (EATC[i][l][m]) * ( (math.exp(R)) / (math.exp(R)-1) )                    

# backward recursive function for the intermediate stages
for i in range(n-2,0,-1):
# stage or time i, backward    
    for l in range(NH):
    # index for existing levee height l as state
        existh = (EXH[i][l])
        # existing levee height l as state 
        if ( i % T ) == 0:
            NHUPm = int(round((Hupmax)/DELTAH+1)) 
            # upgrading time periods, calculate for all possible upgrading heights (UPH[i][m]) = 0
        else:
            NHUPm = 1 
            # none upgrading time periods, upgradh = 0, so could only calculate for the first (UPH[i][1]) = 0
        for m in range(NHUPm):
        # index for upgrading levee height m as decision variable
            upgradh = (UPH[i][m])
            # upgrading levee height m as decision variable, corresponding to existing levee height next stage (EXH[i][l])+(UPH[i][m])
            MINIVALUE = largenumber                
            # initialization of the next stage minimum value for accumulation                             
            logmu = (LOGMU[i])
            logsigma = (LOGSIGMA[i]) 
            # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
            if existh + upgradh <= Hmax + 1e-6:
            # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                (EATC[i][l][m]) = annualtotalcost( existh, upgradh, logmu, logsigma )
                for ll in range(NH):
                # next stage index for existing levee height
                    if abs( (EXH[i+1][ll]) - existh - upgradh ) < 1e-6:
                    # the existing levee height next stage is (EXH[i][l])+(UPH[i][m])                         
                        for mm in range(NHUP):
                        # next stage index for upgrading levee height
                            if (VALUE[i+1][ll][mm]) < MINIVALUE:
                                MINIVALUE = (VALUE[i+1][ll][mm])
            (EMINIVALUE[i][l][m]) = MINIVALUE                
            (VALUE[i][l][m]) = (EATC[i][l][m]) + (EMINIVALUE[i][l][m]) * ( math.exp( -R*1 ) )

# backward recursive function for the (first) starting stage
for i in range(1):
# stage or time i, backward   
    for l in range(l):
    # index for existing levee height l as state
        existh = (EXH[i][l])
        # existing levee height l as state 
        if ( i % T ) == 0:
            NHUPm = int(round((Hupmax)/DELTAH+1)) 
            # upgrading time periods, calculate for all possible upgrading heights (UPH[i][m]) = 0
        else:
            NHUPm = 1 
            # none upgrading time periods, upgradh = 0, so could only calculate for the first (UPH[i][1]) = 0
        for m in range(NHUPm):
        # index for upgrading levee height m as decision variable
            upgradh = (UPH[i][m])
            # upgrading levee height m as decision variable, corresponding to existing levee height next stage (EXH[i][l])+(UPH[i][m])
            MINIVALUE = largenumber                
            # initialization of the next stage minimum value for accumulation                             
            logmu = (LOGMU[i])
            logsigma = (LOGSIGMA[i]) 
            # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
            if existh + upgradh <= Hmax + 1e-6:
            # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                (EATC[i][l][m]) = annualtotalcost( existh, upgradh, logmu, logsigma )
                for ll in range(NH):
                # next stage index for existing levee height
                    if abs( (EXH[i+1][ll]) - existh - upgradh ) < 1e-6:
                    # the existing levee height next stage is (EXH[i][l])+(UPH[i][m])                         
                        for mm in range(NHUP):
                        # next stage index for upgrading levee height
                            if (VALUE[i+1][ll][mm]) < MINIVALUE:
                                MINIVALUE = (VALUE[i+1][ll][mm])
            (EMINIVALUE[i][l][m]) = MINIVALUE                
            (VALUE[i][l][m]) = (EATC[i][l][m]) + (EMINIVALUE[i][l][m]) * ( math.exp( -R*1 ) )
            
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialization for STEP3

OPTVALUE = np.ones(( n, NH )) * largenumber # a very large number
# initialization of the accumulation for the best decisions from all later stages, at time i=1:n, for existing height l=1:NH
# (OPTVALUE[i][l]) i =1:n stage or time; l =1:NH existing levee height
OPTUP = np.zeros(( n, NH )) 
# initialization of minimum upgrading height m, at time i=1:n, for existing height l=1:NH
# (OPTUP[i][l]) i =1:n stage or time; l =1:NH existing levee height
OPTH = np.zeros(( n, NH )) 
# initialization of minimum resulted height l, at time i=1:n, for existing height l=1:NH
# (OPTH[i][l]) i =1:n stage or time; l =1:NH existing levee height


# Initialization for outcome Post-processing
OPTRESULTS=np.zeros((n,5))
# Rows are organized as subsequent stages from 1 to n
# Columns are the index for stage (column 1), existing levee height from previous stage (column 2), OPTUP (column 3), OPTH (column 4), and OPTVALUE (column 5)

for i in range(n):
    Rn = i/NH + 1
    # index for stage
    RNH = ( i%NH ) * DELTAH
    # index for existing levee height at each stage
    for j in range(1):
        (OPTRESULTS[i][j]) = i+1
    for j in range(1,5):
        (OPTRESULTS[i][j]) = largenumber
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 3 Identifying the optimal solutions

# Identifying the optimal results, starting from the first stage
for i in range(1):
# stage or time i                          
    (OPTRESULTS[i][1]) = 0    
    for l in range(1):
    # existing levee height l, starting from H0 = 0                        
        for m in range(NHUP):
        # upgrading levee height m
            if (VALUE[i][l][m]) < (OPTVALUE[i][l]):
                (OPTVALUE[i][l]) = (VALUE[i][l][m])
                (OPTUP[i][l]) = (UPH[i][m])
                (OPTH[i][l]) = (OPTRESULTS[i][1]) + (OPTUP[i][l])        
        (OPTRESULTS[i][2]) = (OPTUP[i][l])
        (OPTRESULTS[i][3]) = (OPTH[i][l])
        (OPTRESULTS[i][4]) = (OPTVALUE[i][l])
    if ( i % T ) == 0:
        print 'Stage:', i+1, 'Exist H:', (OPTRESULTS[i][1]), 'Opt Up H:', (OPTRESULTS[i][2]), 'Result H:', (OPTRESULTS[i][3]), 'Opt Value:', (OPTRESULTS[i][4]) 

for i in range(1,n):
# stage or time i 
    (OPTRESULTS[i][1]) = (OPTRESULTS[i-1][3])                        
    for l in range(NH):
    # existing levee height l, starting from H0 = 0
        if abs( (EXH[i][l]) - (OPTRESULTS[i][1]) ) < 1e-6:                        
            for m in range(NHUP):
                # upgrading levee height m
                if (VALUE[i][l][m]) < (OPTVALUE[i][l]):
                    (OPTVALUE[i][l]) = (VALUE[i][l][m])
                    (OPTUP[i][l]) = (UPH[i][m])
                    (OPTH[i][l]) = (OPTRESULTS[i][1]) + (OPTUP[i][l])
            (OPTRESULTS[i][2]) = (OPTUP[i][l])
            (OPTRESULTS[i][3]) = (OPTH[i][l])
            (OPTRESULTS[i][4]) = (OPTVALUE[i][l])
    if ( i % T ) == 0:
        print 'Stage:', i+1, 'Exist H:', (OPTRESULTS[i][1]), 'Opt Up H:', (OPTRESULTS[i][2]), 'Result H:', (OPTRESULTS[i][3]), 'Opt Value:', (OPTRESULTS[i][4]) 
                                
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
#print "Optimal overall cost", OPTVALUE
#print "Optimal upgrading height", OPTUP
#print "Optimal resulting height", OPTH


with open("DP20_5_1.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(OPTRESULTS)
            
# NUMBERS in csv file name represent year interval, number of interval and number of climate scenario        
        
end = time.time()
print "Computation time", (end - start)       
        
        
        
        
        
        
        
        
        
        
        
        
        
