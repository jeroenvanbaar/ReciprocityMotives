import os
import numpy as np
import pandas as pd

def MP_model(inv, mult, baseMult, exp, theta, phi):
    inv = float(inv); mult = float(mult); baseMult = float(baseMult);
    exp = float(exp); theta = float(theta); phi = float(phi);
    # exp = .5*(10 - inv + inv*baseMult)-(10-inv);

    totalAmt = inv*mult
    choiceOpt = np.arange(0,totalAmt+1) # Only integers in strategy space (but not further discretized)
    
    own = totalAmt-choiceOpt
    other = 10 - inv + choiceOpt
    ownShare = own/totalAmt # Should be totalAmt

    guilt = np.square(np.maximum((exp-choiceOpt)/(inv*baseMult),0))
    inequity = np.square(own/(own+other) - .5)

    utility = theta*ownShare - (1-theta)*np.minimum(guilt+phi, inequity-phi)
    
    return choiceOpt[np.where(utility == np.max(utility))[0][0]]

def MP_model_ppSOE(inv, mult, baseMult, exp, theta, phi):
    # 'pre-programmed Second-Order Expectations'
    inv = float(inv); mult = float(mult); baseMult = float(baseMult);
    exp = float(0.5*baseMult*inv); theta = float(theta); phi = float(phi);

    totalAmt = inv*mult
    choiceOpt = np.arange(0,totalAmt+1) # Only integers in strategy space (but not further discretized)
    
    own = totalAmt-choiceOpt
    other = 10 - inv + choiceOpt
    ownShare = own/totalAmt # Should be totalAmt

    guilt = np.square(np.maximum((exp-choiceOpt)/(inv*baseMult),0))
    inequity = np.square(own/(own+other) - .5)

    utility = theta*ownShare - (1-theta)*np.minimum(guilt+phi, inequity-phi)
    
    return choiceOpt[np.where(utility == np.max(utility))[0][0]]

def IA_model(inv, mult, baseMult, exp, theta, phi):
    inv = float(inv); mult = float(mult); baseMult = float(baseMult);
    exp = float(exp); theta = float(theta); phi = float(phi);

    totalAmt = inv*mult
    choiceOpt = np.arange(0,totalAmt+1)
    
    own = totalAmt-choiceOpt
    other = 10 - inv + choiceOpt

    inequity = np.square(own/(own+other) - .5)

    utility = own - theta*inequity

    return choiceOpt[np.where(utility == np.max(utility))[0][0]]

def GA_model(inv, mult, baseMult, exp, theta, phi):
    inv = float(inv); mult = float(mult); baseMult = float(baseMult);
    exp = float(exp); theta = float(theta); phi = float(phi);
    # exp = .5*(10 - inv + inv*baseMult)-(10-inv);

    totalAmt = inv*mult
    choiceOpt = np.arange(0,totalAmt+1)
    
    guilt = np.square(np.maximum((exp-choiceOpt)/(inv*baseMult),0))
    
    own = totalAmt-choiceOpt

    utility = own - theta*guilt

    return choiceOpt[np.where(utility == np.max(utility))[0][0]]

def GA_model_ppSOE(inv, mult, baseMult, exp, theta, phi):
    inv = float(inv); mult = float(mult); baseMult = float(baseMult);
    exp = float(0.5*baseMult*inv); theta = float(theta); phi = float(phi);

    totalAmt = inv*mult
    choiceOpt = np.arange(0,totalAmt+1)
    
    guilt = np.square(np.maximum((exp-choiceOpt)/(inv*baseMult),0))
    
    own = totalAmt-choiceOpt

    utility = own - theta*guilt

    return choiceOpt[np.where(utility == np.max(utility))[0][0]]

def GR_model(inv, mult, baseMult, exp, theta, phi):
    inv = float(inv); mult = float(mult); baseMult = float(baseMult);
    exp = float(exp); theta = float(theta); phi = float(phi);
    return 0

def hybrid_model(inv,mult,baseMult,exp,theta,phi):
    inv = float(inv); mult = float(mult); baseMult = float(baseMult);
    exp = float(exp); theta = float(theta); phi = float(phi);

    totalAmt = inv*mult
    choiceOpt = np.arange(0,totalAmt+1)

    own = totalAmt-choiceOpt
    ownShare = own/totalAmt
    other = 10 - inv + choiceOpt

    inequity = np.square(np.maximum(own/(own+other) - .5,0))
    guilt = np.square(np.maximum((exp-choiceOpt)/(inv*baseMult),0))

    comparison = ((guilt+phi)-(inequity-phi))>=0 # Indexes where inequity is smallest
    socPref = np.zeros([1,len(choiceOpt)])[0]
    socPref[comparison==True] = inequity[comparison==True]
    socPref[comparison==False] = guilt[comparison==False]

    exponent = .3
    thetaTrans = theta**exponent/((.2**exponent)/.2)
    utility = own - (.2-thetaTrans)*3000*socPref
    
    choice = choiceOpt[np.where(utility == np.max(utility))[0][0]]

    return choice
