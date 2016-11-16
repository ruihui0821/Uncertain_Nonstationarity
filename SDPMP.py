import time
start = time.time()

import numpy as np
import math  # This will import math module
from scipy.stats import lognorm # A lognormal continuous random variable. lognorm.pdf(x, s) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)
# Specifically, lognorm.pdf(x, s, loc, scale) is identically equivalent to lognorm.pdf(y, s) / scale with y = (x - loc) / scale.
from scipy.integrate import quad
import csv

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

largenumber = 1e+20 # a very large number for calculation purpose

# Planning time periods
T = 10 # the year interval that a levee to be upgrading (yrs)
NT = 10 # the number of intervals in a levee's lifetime
n = int(NT * T) # a levee total lifetime (yrs)

# Levee design standard
H0 = 0.0 # The initial levee height (meter)
DELTAH = 0.1 # Step size of levee height (meter)
Hmin = 0.0 # The minimum levee height (meter)
Hmax = 10.0 # The maximum levee height (meter)
Hupmax = 10.0 # The maximum upgrading levee height (meter)

# Financial Parameters
DC = 10.0 # Flood Damage Cost, assuming simultaneous failures on two riversides with a constant value (million $)
Cland = 1.0 # Price of land ($/m**2) 
Csoil = 30.0 # Soil compaction cost or construction cost per unit levee material ($/m**3)
Cadjust =1.3 # Soft Cost Multiplier considering management
R = 0.05 # Real (inflation-adjusted) discount rate

# Stationary Annual Flow Parameters
mu0 = 100.0 # Mean annual average peak flow (m**3/s)
sigma0 = 50.0 # Standard deviation of annual average peak flow

# climate scenarios
NMU = 5 # number of different changing mean
NSIGMA = 3 # number of different changing standard deviation
NA = NMU * NSIGMA # number of climate scenarios

muslope = np.zeros(NMU) 
# initialization of variation of possible annual average peak flow
for i in range(NMU):   
    muslope[i] = 0.25/100 * i    
sigmaslope = np.zeros(NSIGMA) 
# initialization of variation of possible annual standard deviation of peak flow
for i in range(NSIGMA):   
    sigmaslope[i] = 0.25/100 * i     

PA = np.zeros(NA) 
# initialization of probability of each possible climate scenarios, the same
MU = np.zeros((n,NA))
# initialization of annual average peak flow for n years and NA climate scenarios
SIGMA = np.zeros((n,NA))
# initialization of annual standard deviation of peak flow for n years and NA climate scenarios
LOGMU = np.zeros((n,NA))
# initialization of location parameter of annual peak flow for n years and NA climate scenarios
LOGSIGMA = np.zeros((n,NA))
# initialization of scale parameter of annual peak flow for n years and NA climate scenarios
# On a logarithmic scale, {\displaystyle \mu } \mu  and {\displaystyle \sigma } \sigma  can be called the location parameter and the scale parameter, respectively.
for i in range(n):
# stage or time i        
    for k in range(NA):
    # a climate scenario or a combination of mean annual peak flow MU and standard deviation SIGMA
        PA[k] = (1.0/NA)
        ki = k % NMU
        MU[i,k] = mu0 + muslope[ki] * mu0 * (i + 1) 
        # annual average peak flow for time i and climate scenario j
        ik = k / NMU
        SIGMA[i,k] = sigma0 + sigmaslope[ik] * sigma0 * (i + 1) 
        # annual average peak flow for time i and climate scenario j
        LOGMU[i,k] = math.log( ( (MU[i,k])**2) / math.sqrt( SIGMA[i,k] + (MU[i,k])**2 ) )
        LOGSIGMA[i,k] = math.sqrt( math.log( SIGMA[i,k] / ( (MU[i,k])**2 ) + 1 ) )


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
L = 3000.0 # Total levee length (meter)
 
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
    PQ = lognorm.pdf(Q, intemu, intesigma) # # Probability of Flow Q (PDF)
    if QCMAX == QCMIN:
        PNOVER = 1
    else:
        PNOVER = ( Q - QCMIN )/( QCMAX - QCMIN )
    return PQ*PNOVER

def intermediatefailure(intemu, intesigma, inteheight):
    Htop = Htoe + inteheight
    QCMAX = flow( Htop )
    FINTERMEDIATE = quad(integrand, QCMIN, QCMAX, args=(intemu, intesigma, inteheight))[0]
    return FINTERMEDIATE

##################  Overflow failure probability
def overtopfailure(overmu, oversigma, overheight):
    Htop = Htoe + overheight
    OVERFLOW = flow(Htop) # Overtopping Flow
    FNOVER = lognorm.cdf(OVERFLOW, overmu, oversigma)
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
# SDP Program with Markov Process
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#STEP 1 Calculation of all conditional probabilities for possible observed climate scenarios at previous stage

PMUA = np.zeros((( n, NA, NA )))
# initialization of the conditional probability of a mean flow MU given a climate scenario, at time i=1:n
# (PMUA[i][k][s]) i =1:n stage or time; k =1:NA a mean flow MU; s =1:NA a climate scenario
for i in range(n):
# stage or time i             
    for k in range(NA):
    # the climate scenario or MU
        s = 0
        # a climate scenario
        PMUA0 = lognorm.pdf( (LOGMU[i][k]), (LOGMU[i][s]), (LOGSIGMA[i][s]) ) 
        while s < NA - 1:
            s = s + 1
            PMUA0 = PMUA0 + lognorm.pdf( (LOGMU[i][k]), (LOGMU[i][s]), (LOGSIGMA[i][s]) )
        for s in range(NA):
        # a climate scenario
            (PMUA[i][k][s]) = lognorm.pdf( (LOGMU[i][k]), (LOGMU[i][s]), (LOGSIGMA[i][s]) ) / PMUA0
            # lognorm.pdf( (LOGMU[i][k]), (LOGMU[i][s]), (LOGSIGMA[i][s]) ) is the probability of flow (LOGMU[i][k]) given mean and variation

PAMU = np.zeros((( n, NA, NA )))
# initialization of the conditional probability of a current climate scenario (k=1:NA) given an observed climate in a previous climate scenario (j=1:NA), at time i=1:n
# (PAMU[i][j][k]) i =1:n stage or time; j =1:NA an observed previous climate scenario; k =1:NA a current climate scenario

# the (first) starting stage climate scenarios info.
for i in range(1):
# stage or time i        
    for j in range(NA):
    # an observed previous climate scenario       
        for k in range(NA):
        # a current climate scenario
            (PAMU[i][j][k]) = (PA[k])
            
# the rest stages climate scenarios info.
for i in range(1,n):
# stage or time i             
    for j in range(NA):
    # an observed previous climate scenario         
        for k in range(NA):
        # a current climate scenario      
            s = 0
            # a cliamte scenario
            PPMU = (PMUA[i-1][j][s]) * (PA[s])
            while s < NA - 1:
                s = s +1
                PPMU = PPMU + (PMUA[i-1][j][s]) * (PA[s])
                # calculation for the probability of a previous observed climate scenario, PPMU
            (PAMU[i][j][k]) = (PA[k]) * (PMUA[i-1][j][k]) / PPMU
            # the conditional probability of a climate scenario given an observed climate scenario
                
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#STEP 2 Calculation of the backward SDP value functions

EATC = np.ones(((( n, NA, NH, NHUP )))) * largenumber
# initialization of the expected annual total cost for upgrading height m=1:NHUP and existing height l=1:NH; given an observed previous climate scenario (j=1:NA), at time i=1:n 
# (EATC[i][j][l][m]) i =1:n stage or time; j =1:NA an observed previous climate scenario; l =1:NH existing levee height; m =1:NHUP upgrading levee height
VALUE = np.ones(((( n, NA, NH, NHUP )))) * largenumber # a very large number
# initialization of the accumulation for the decisions from all later stages, 
# for upgrading height m=1:NHUP and existing height l=1:NH, given an observed previous climate scenario (j=1:NA), at time i=1:n
# (VALUE[i][j][l][m]) i =1:n stage or time; j =1:NA an observed previous climate scenario; l =1:NH existing levee height; m =1:NHUP upgrading levee height
EMINIVALUE = np.ones(((( n, NA, NH, NHUP )))) * largenumber # a very large number
# initialization of the expection of the next stage minimum value for accumulation  
# given an observed previous climate scenario (j=1:NA), at time i=1:n, with current existing height l=1:NH and upgrading height m=1:NHUP
# (EMINIVALUE[i][j][l][m]) i =1:n stage or time; j =1:NA an observed previous climate scenario; l =1:NH existing levee height; m =1:NHUP upgrading levee height

# backward recursive function for the last stage
for i in range(n-1,n):
# stage or time i, backward index(n-1-i), starting from the last stage index(n-1)           
    for j in range(NA):
    # an observed previous climate scenario    
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
                k = 0
                # current climate scenario
                logmu = (LOGMU[i][k])
                logsigma = (LOGSIGMA[i][k])                
                if existh + upgradh <= Hmax + 1e-6:
                # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                    (EATC[i][j][l][m]) = annualtotalcost( existh, upgradh, logmu, logsigma ) * ( 1/((1+R)**i) ) * (PAMU[i][j][k])
                while k < NA - 1:
                    k = k + 1
                    logmu = (LOGMU[i][k])
                    logsigma = (LOGSIGMA[i][k])
                    if existh + upgradh <= Hmax + 1e-6:
                        (EATC[i][j][l][m]) = (EATC[i][j][l][m]) + annualtotalcost( existh, upgradh, logmu, logsigma ) * ( 1/((1+R)**i) ) * (PAMU[i][j][k])
                        # adding up for expected annual total cost over all possible climate scenarios at stage i
                (EMINIVALUE[i][j][l][m]) = 0 # No future expected value to be added
                (VALUE[i][j][l][m]) = ( (EATC[i][j][l][m]) + (EMINIVALUE[i][j][l][m]) ) * (math.exp(R*(i+1))/(math.exp(R*(i+1))-1))                   

# backward recursive function for the intermediate stages
for i in range(n-2,0,-1):
# stage or time i, backward 
    for j in range(NA):
    # an observed previous climate scenario   
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
                k = 0
                # current climate scenario
                MINIVALUE = largenumber                
                # initialization of the next stage minimum value for accumulation                             
                logmu = (LOGMU[i][k])
                logsigma = (LOGSIGMA[i][k])
                # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                if existh + upgradh <= Hmax + 1e-6:
                # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                    (EATC[i][j][l][m]) = annualtotalcost( existh, upgradh, logmu, logsigma ) * ( 1/((1+R)**i) ) * (PAMU[i][j][k])
                    for ll in range(NH):
                    # next stage index for existing levee height
                        if abs( (EXH[i+1][ll]) - existh - upgradh ) < 1e-6:
                        # the existing levee height next stage is (EXH[i][l])+(UPH[i][m])                         
                            for mm in range(NHUP):
                            # next stage index for upgrading levee height
                                if (VALUE[i+1][k][ll][mm]) < MINIVALUE:
                                    MINIVALUE = (VALUE[i+1][k][ll][mm])
                (EMINIVALUE[i][j][l][m]) = MINIVALUE * (PAMU[i][j][k])                
                while k < NA - 1:
                    k = k + 1
                    MINIVALUE = largenumber                
                    # initialization of the next stage minimum value for accumulation
                    logmu = (LOGMU[i][k])
                    logsigma = (LOGSIGMA[i][k])
                    # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                    if existh + upgradh <= Hmax + 1e-6:
                    # expected annual total cost only exists when existing + upgrading height is no greater than maximum height   
                        (EATC[i][j][l][m]) = (EATC[i][j][l][m]) + annualtotalcost( existh, upgradh, logmu, logsigma ) * ( 1/((1+R)**i) ) * (PAMU[i][j][k])
                        # adding up for expected annual total cost over all possible climate scenarios at stage i
                        for ll in range(NH):
                        # next stage index for existing levee height
                            if abs( (EXH[i+1][ll]) - existh - upgradh ) < 1e-6:
                            # the existing levee height next stage is (EXH[i][l])+(UPH[i][m]) 
                                for mm in range(NHUP):
                                # next stage index for upgrading levee height
                                    if (VALUE[i+1][k][ll][mm]) < MINIVALUE:
                                        MINIVALUE = (VALUE[i+1][k][ll][mm])
                    (EMINIVALUE[i][j][l][m]) = (EMINIVALUE[i][j][l][m]) + MINIVALUE * (PAMU[i][j][k])  
                    # adding up for expected minimum future accumulated value at next stage i+1 over all possible current climate scenarios 
                (VALUE[i][j][l][m]) = (EATC[i][j][l][m]) + (EMINIVALUE[i][j][l][m])

# backward recursive function for the (first) starting stage
for i in range(1):
# stage or time i, backward   
    for j in range(NA):
    # an observed previous climate scenario       
        for l in range(1):
        # index for existing levee height l as state
            existh = (EXH[i][l])
            # existing levee height l as state 
            for m in range(NHUP):
            # index for upgrading levee height m as decision variable
                upgradh = (UPH[i][m])
                # upgrading levee height m as decision variable, corresponding to existing levee height next stage (EXH[i][l])+(UPH[i][m])
                k = 0
                # current climate scenario
                MINIVALUE = largenumber                
                # initialization of the next stage minimum value for accumulation                             
                logmu = (LOGMU[i][k])
                logsigma = (LOGSIGMA[i][k])
                # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                if existh + upgradh <= Hmax + 1e-6:
                # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                    (EATC[i][j][l][m]) = annualtotalcost( existh, upgradh, logmu, logsigma ) * ( 1/((1+R)**i) ) * (PAMU[i][j][k])
                    for ll in range(NH):
                    # next stage index for existing levee height
                        if abs( (EXH[i+1][ll]) - existh - upgradh ) < 1e-6:
                        # the existing levee height next stage is (EXH[i][l])+(UPH[i][m])                         
                            for mm in range(NHUP):
                            # next stage index for upgrading levee height
                                if (VALUE[i+1][k][ll][mm]) < MINIVALUE:
                                    MINIVALUE = (VALUE[i+1][k][ll][mm])
                (EMINIVALUE[i][j][l][m]) = MINIVALUE * (PAMU[i][j][k])                
                while k < NA - 1:
                    k = k + 1
                    MINIVALUE = largenumber                
                    # initialization of the next stage minimum value for accumulation
                    logmu = (LOGMU[i][k])
                    logsigma = (LOGSIGMA[i][k])
                    # expected annual total cost only exists when existing + upgrading height is no greater than maximum height
                    if existh + upgradh <= Hmax + 1e-6:
                    # expected annual total cost only exists when existing + upgrading height is no greater than maximum height   
                        (EATC[i][j][l][m]) = (EATC[i][j][l][m]) + annualtotalcost( existh, upgradh, logmu, logsigma ) * ( 1/((1+R)**i) ) * (PAMU[i][j][k])
                        # adding up for expected annual total cost over all possible climate scenarios at stage i
                        for ll in range(NH):
                        # next stage index for existing levee height
                            if abs( (EXH[i+1][ll]) - existh - upgradh ) < 1e-6:
                            # the existing levee height next stage is (EXH[i][l])+(UPH[i][m]) 
                                for mm in range(NHUP):
                                # next stage index for upgrading levee height
                                    if (VALUE[i+1][k][ll][mm]) < MINIVALUE:
                                        MINIVALUE = (VALUE[i+1][k][ll][mm])
                    (EMINIVALUE[i][j][l][m]) = (EMINIVALUE[i][j][l][m]) + MINIVALUE * (PAMU[i][j][k])  
                    # adding up for expected minimum future accumulated value at next stage i+1 over all possible current climate scenarios 
                (VALUE[i][j][l][m]) = (EATC[i][j][l][m]) + (EMINIVALUE[i][j][l][m])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialization for STEP3

OPTVALUE = np.ones((( n, NH, NA ))) * largenumber # a very large number
# initialization of the accumulation for the best decisions from all later stages, at time i=1:n, for existing height l=1:NH, given an observed previous climate scenario (j=1:NA)
# (OPTVALUE[i][l][j]) i =1:n stage or time; l =1:NH existing levee height; j =1:NA an observed previous climate scenario
OPTUP = np.zeros((( n, NH, NA ))) 
# initialization of minimum upgrading height m, at time i=1:n, for existing height l=1:NH, given an observed previous climate scenario (j=1:NA)
# (OPTUP[i][l][j]) i =1:n stage or time; l =1:NH existing levee height; j =1:NA an observed previous climate scenario
OPTH = np.zeros((( n, NH, NA ))) 
# initialization of minimum resulted height l, at time i=1:n, for existing height l=1:NH, given an observed previous climate scenario (j=1:NA)
# (OPTH[i][l][j]) i =1:n stage or time; l =1:NH existing levee height; j =1:NA an observed previous climate scenario


# Initialization for outcome Post-processing
OPTRESULTS=np.zeros((n*NH,3*NA+2))
# Rows are organized as subsequent stages from 1 to n, with every possible existing levee height NH in each stage
# Columns are organized as the index for stage (column 1), index for existing levee height (column 2), 
# OPTUP for each climate scenario (column 3~NA+2), OPTH for each climate scenario NA (column NA+3~2*NA+2), and OPTVALUE for each climate scenario (column 2*NA+3~3*NA+2)

for i in range(n*NH):
    Rn = i/NH + 1
    # index for stage
    RNH = ( i%NH ) * DELTAH
    # index for existing levee height at each stage
    for j in range(1):
        (OPTRESULTS[i][j]) = Rn
    for j in range(1,2):
        (OPTRESULTS[i][j]) = RNH
    for j in range(2*NA+2,3*NA+2):
        (OPTRESULTS[i][j]) = largenumber
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# STEP 3 Identifying the optimal solutions

# Identifying the optimal results, starting from the first stage
for i in range(n):
# stage or time i                          
    for l in range(NH):
    # existing levee height l, starting from H0 = 0
        for j in range(NA):
            # an observed previous climate scenario, none for the first stage                          
            for m in range(NHUP):
            # upgrading levee height m
                if (VALUE[i][j][l][m]) < (OPTVALUE[i][l][j]):
                    (OPTVALUE[i][l][j]) = (VALUE[i][j][l][m])
                    (OPTUP[i][l][j]) = (UPH[i][m])
                    (OPTH[i][l][j]) = (EXH[i][l]) + (OPTUP[i][l][j])
            (OPTRESULTS[i*NH+l][j+2]) = (OPTUP[i][l][j])
            (OPTRESULTS[i*NH+l][j+NA+2]) = (OPTH[i][l][j])
            (OPTRESULTS[i*NH+l][j+2*NA+2]) = (OPTVALUE[i][l][j])
            if ( i % T ) == 0:
                print 'Stage:', i+1, 'Exist H:', (EXH[i][l]), 'Ob Climate:', j+1, 'Opt Up H:', (OPTUP[i][l][j]), 'Result H:', (OPTH[i][l][j]), 'Opt Value:', (OPTVALUE[i][l][j]) 
                
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
#print "Optimal overall cost", OPTVALUE
#print "Optimal upgrading height", OPTUP
#print "Optimal resulting height", OPTH


with open("SDPMP10_10_10.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(OPTRESULTS)

# NUMBERS in csv file name represent year interval, number of interval and number of climate scenarios  
        
end = time.time()
print "Computation time", (end - start)       
        
        
        
        
        
        
        
        
        
        
        
        
        
