import pandas as pd
import numpy as np
import choiceModels

def simulate_data_study1(model = 'MP',theta=-99, phi=-99,multipliers = [2,4,6],investments=np.arange(1,11)):
    modelDict = {'MP':choiceModels.MP_model,'IA':choiceModels.IA_model,
                 'GA':choiceModels.GA_model,'GR':choiceModels.GR_model}
    baseMult = multipliers[1]
    simulations = pd.DataFrame(columns=['model','Investment','Multiplier','Believed multiplier',
    	'Expectation','Amount returned','theta','phi'])
    for mult in multipliers:
        for inv in investments:
            exp = baseMult/2*inv
            ret = modelDict[model](inv,mult,baseMult,exp,theta,phi)
            simulations = simulations.append(pd.DataFrame(
                [[model,inv,'x%i'%mult,'x%i'%baseMult,exp,ret,theta,phi]],
                columns=simulations.columns))
    return simulations

def simulate_data_study2(model = 'MP',theta=-99, phi=-99,multipliers = [2,4,6],investments=np.arange(1,11)):
    modelDict = {'MP':choiceModels.MP_model,'IA':choiceModels.IA_model,
                 'GA':choiceModels.GA_model,'GR':choiceModels.GR_model}
    mult = multipliers[1]
    simulations = pd.DataFrame(columns=['model','Investment','Multiplier','Believed multiplier',
    	'Expectation','Amount returned','theta','phi'])
    for belMult in multipliers:
        for inv in investments:
            exp = belMult/2*inv
            ret = modelDict[model](inv,mult,belMult,exp,theta,phi)
            simulations = simulations.append(pd.DataFrame(
                [[model,inv,'x%i'%mult,'x%i'%belMult,exp,ret,theta,phi]],
                columns=simulations.columns))
    return simulations