# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:24:38 2023

@author: william Lewin
"""

#imports
import numpy as np
import matplotlib.pyplot as plt
import uproot
import pandas as pd
from scipy.optimize import curve_fit
from PhiCalculator import Phi_Loc_Est

def main():
    plt.style.use('default')
    
    
    # Open root file
    #path = 'C:/Users/willi/Documents/Uni Work/Year 3/Group Project/PathReconstruction10.root'
    #Tree = uproot.open(path)
    
    #TestEventLocalisation('X-Ray Source(N3, N5)')
    
    #Open Multiple root files
    mainDirectory = 'C:/Users/willi/Documents/Uni Work/Year 3/Group Project/'
    currentSubDirectory = 'GammaData' # output
    fileNameCore = 'output_'
    fileNumber = range(100)
    paths = GetMultipleFilePaths(mainDirectory, currentSubDirectory, fileNameCore, fileNumber)
    
    #Currently code is plott
    PathReconstruction(paths)
    
# Event Localisation

# Algorithm Is used to pass all the path 
def EventLocalisation(_paths):
    X = []
    Y = []
    deviation = []
    
    
    # Get phi of the far field anodes
    anodePhis = np.linspace(np.pi/10, np.pi/10 - 2* np.pi, 5, endpoint = False)
    
    # Cast last two phis (< -pi) into positive phi 
    anodePhis[3:5] = anodePhis[3:5]+ 2 * np.pi
        
    # For each file
    for i in _paths:
        tree = uproot.open(i)
        events = tree['simul;1/InitialParticleType'].array(library = 'np').shape[0]
        
        #For each event
        for j in range(events):
            
            #Reject unsucessful events and events with secondary emmissions
            if(EventRejection(tree, j) == True):
                
                #Get amplitudes and risetimes
                l = 0
                anodeAmplitudes = np.zeros(11)
                anodeRisetimes = np.zeros(11)
                for k in GetAnodes():
                    anodeAmplitudes[l] = tree[f'simul;1/{k}Amplitude'].array(library = 'np')[j]
                    anodeRisetimes[l] = tree[f'simul;1/{k}Risetime'].array(library = 'np')[j]
                    l = l + 1

                l = 0

                #Apply algorithm
                pos = LocalisationAlgorithm(anodeAmplitudes, anodeRisetimes)
                if pos != None:
                    
                    #Extact the data you need here
                    
                    
                    if pos[0] != None:
                        
                        # Phi:
                        phi = np.average(tree['simul;1/InitElecPhi'].array(library = 'np')[j])
                        phiDelta = (phi - GetNearAnodePhis()[pos[2]-6])* pos[1]
                        if phiDelta < -5:
                            phiDelta = phiDelta + 2 * np.pi
                        
                        X.append(phiDelta)
                        Y.append(pos[0] - np.average(tree['simul;1/InitElecCosTheta'].array(library = 'np')[j]))
                        
                        # Cos fit
                        #X.append(pos[0])
                        #Y.append(np.average(tree['simul;1/InitElecCosTheta'].array(library = 'np')[j]))
    
    
    #Currently being used to fit data
    fit = curve_fit(Cubic, X, Y)[0]
    print(fit)
    t = np.linspace(0, 1)
    plt.plot(t, Cubic(t, fit[0], fit[1], fit[2], fit[3]), color = 'r')
    plt.scatter(X, Y)
    
# Actual Algorithm, takes amplitudes of anodes (F0-5, N1-5) and rise times (F0-5, N1-5)
def LocalisationAlgorithm(anodeAmplitudes, anodeRisetimes):
    # Attempts to return the position in the form (r, cosTheta Phi)
    
    #Find peak anode and select the corresponding procedure
    peakAnode = np.argmax(anodeAmplitudes)
    
    radius = LocalisationRadius(anodeRisetimes[peakAnode])

    #More 
    phi = Phi_Loc_Est(anodeAmplitudes, radius)
    
    #Significance
    minAmplitude = 0.004
    
    #Reject event if peak signal is too low
    if(anodeAmplitudes[peakAnode] < minAmplitude):
        print('Insignificant signal, event recjected')
        return
    
    # Catch multiple depositions
    
    if(peakAnode == 0):  # F0 is peak anode
        angles = LocalisationAtF0(anodeAmplitudes[1:6], anodeAmplitudes[0])
        return radius, angles[0], angles[1]
    
    elif(peakAnode < 6):  # Peak anode is Far field
        angles = LocalisationFarField(anodeAmplitudes, peakAnode, phi[1])
        return radius, angles[0], angles [1] 
        
    else:  # Peak anode is near field
        angles = LocalisationNearField(anodeAmplitudes, peakAnode, phi[1])
        return radius, angles[0], angles [1]
  
# F0 component of algorithm
def LocalisationAtF0(amplitudesFF, F0Amplitude):

    anodePhis = GetFarAnodePhis()
    
    # Find largest and second largest amplitudes in the far field anodes
    maxAnodeFF = np.argmax(amplitudesFF)
    
    # Define a minimum amplitude required for further localisation
    minAmplitude = 0.002
    
    if(amplitudesFF[maxAnodeFF] < minAmplitude):
        # Event occured close to F0, with no significant signals in other anodes
        return 1, 0
        
    #Determine second highest amplitude in far field 
    removeAmplitudes = amplitudesFF.copy()  # Create duplicate array
    removeAmplitudes[maxAnodeFF] = np.min(removeAmplitudes)  # Set the max value to the minimum value in array to remove it
    secondMaxAnodeFF = np.argmax(removeAmplitudes)  # Select highest remaining anode index
    
    #Create different processes if only the second amplitude is large enough
    if(amplitudesFF[secondMaxAnodeFF] < minAmplitude):
        #Only max in far field is significant
        
        #Determine Cos values here
        
        ratio = amplitudesFF[maxAnodeFF]/F0Amplitude
        
        #Cos theta fit of F0 data with one far field anode
        cosTheta = Cubic(ratio, -0.162226, 0.333851, -0.251812, 0.902932)
        return cosTheta, anodePhis[maxAnodeFF]
        
    else:
        # 2 Significant signals in far field. Better phi localisation possible
        
        # Phi: 
        ratio = amplitudesFF[secondMaxAnodeFF]/amplitudesFF[maxAnodeFF]
        
        deltaPhi = Cubic(ratio, 1.161334, -2.008821, 1.11336, 0.392574)
        
        direction = 1
        if (secondMaxAnodeFF > maxAnodeFF) or (maxAnodeFF == 4 and secondMaxAnodeFF == 0):
            direction = -1
        
        phi = anodePhis[maxAnodeFF] + direction * deltaPhi
        
        # CosTheta:
        ratio = (amplitudesFF[secondMaxAnodeFF] + amplitudesFF[maxAnodeFF])/F0Amplitude
        
        cosTheta = Cubic(ratio, -0.022546, 0.096989, -0.155745, 0.87384)
        
        cosThetaPhiCorrection = Cubic(deltaPhi, -4.5427645, 7.512485, -4.152949, 0.766219)
        return cosTheta + cosThetaPhiCorrection, phi
    
def LocalisationFarField(amplitudes, peakAnode, phi_err):
    
    # Determining Phi first:
    peakPhiAnode = peakAnode - 1
    if peakPhiAnode == 0:
        peakPhiAnode = 5
    
    anodeClockwise = peakAnode + 1
    if anodeClockwise == 6:
        anodeClockwise = 1
        
    # Check which anode either side is greatest:
    direction = direction = np.sign(phi_err) #1
    
    if amplitudes[anodeClockwise] > amplitudes[peakPhiAnode]:
        peakPhiAnode = anodeClockwise
        #direction = -1
    
    #Check this side anode has a significant signal
    minAmplitude = 0.002
    if (amplitudes[peakPhiAnode] > minAmplitude):
        # Phi result is refinable
        ratio = amplitudes[peakPhiAnode]/amplitudes[peakAnode]
        #deltaPhi = #Cubic(ratio, 0.72138513, -1.28488927, 0.7873654, 0.42802945)
        
        #Move onto CosTheta
        return LocalisationFarFieldCosTheta(amplitudes, peakAnode, abs(phi_err)), GetFarAnodePhis()[peakAnode-1] + phi_err
    
    else:
        #No clear phi data:
        return LocalisationFarFieldCosTheta(amplitudes, peakAnode, 0), GetFarAnodePhis()[peakAnode-1], 0
        
def LocalisationFarFieldCosTheta(amplitudes, peakAnode, deltaPhi):
    minAmplitude = 0.002
    #CosTheta determination:
    NearFieldAnodeClockwise = peakAnode+6
    NearFieldAnodeAntiClockwise = peakAnode + 5
    if NearFieldAnodeClockwise == 11:
        NearFieldAnodeClockwise = 6 
    
    # Get Near field signal:
    nearFieldSignal = amplitudes[NearFieldAnodeAntiClockwise] + amplitudes[NearFieldAnodeClockwise]
    if(amplitudes[0] > nearFieldSignal):
        #F0 stronger than near field signal
        
        if(amplitudes[0] > minAmplitude):
            ratio = amplitudes[0]/amplitudes[peakAnode]
            if deltaPhi != 0:
                # F0 signal significant, phi information available
                cosTheta = Cubic(ratio, 0.42020661, -0.77474954, 0.48278955, 0.66922289)
                return cosTheta - Cubic(deltaPhi, -1.96531409, 2.77319341, -1.13202557, 0.12093198)
            else: 
                # F0 signal significant, no phi information
                return Cubic(ratio, 0.54196717, -0.95887189, 0.55647813, 0.70037675)
        else:
            # No signals on anything but main anode
            return 0.432
        
    
    else:
        if(nearFieldSignal > minAmplitude):
            if(deltaPhi != 0):
                # Near field and phi data
                ratio = nearFieldSignal/amplitudes[peakAnode]
                cosTheta = Quartic(ratio, 1.54407468, -3.25884271, 2.53285846, -0.94692889, 0.16076826)
                return cosTheta - Cubic(deltaPhi, 12.54314877, -19.29557737, 9.2912493, -1.37343871)
            else:
                # Near field signal significant, no phi information
                ratio = nearFieldSignal /amplitudes[peakAnode]
                return Cubic(ratio, -0.21905159, 0.50360696, -0.52390729, -0.07262517)
        else:
            # No signals on anything but main anoder
            return 0.432

def LocalisationNearField(amplitudes, peakAnode, phi_err):
    
    # Determining Phi first:
    peakPhiAnode = peakAnode - 1
    if peakPhiAnode == 5:
        peakPhiAnode = 10
    
    anodeClockwise = peakAnode + 1
    if anodeClockwise == 11:
        anodeClockwise = 6
        
    # Check which anode either side is greatest:
    direction = 1
        
    if amplitudes[anodeClockwise] > amplitudes[peakPhiAnode]:
        peakPhiAnode = anodeClockwise
        direction = -1
    
    # Get anodes above in far field:
    anodeFarAntiClockwise = peakAnode - 6
    anodeFarClockwise = peakAnode - 5
    if anodeFarAntiClockwise == 0:
        anodeFarAntiClockwise = 5
    
    #Check the phi anode has a significant signal
    minAmplitude = 0.002
    if (amplitudes[peakPhiAnode] > minAmplitude):
        # Phi result is refinable
        ratio = amplitudes[peakPhiAnode]/amplitudes[peakAnode]
        #deltaPhi = Quartic(ratio, -1.61205301, 3.67296451, -2.98130395, 1.13861822, 0.39322815)
        
        #CosTheta
        anodeFarField = anodeFarAntiClockwise
        if direction == -1:
            anodeFarField = anodeFarClockwise
        
        ratio = amplitudes[anodeFarField]/amplitudes[peakAnode]
        
        # Exponential decay provided a better fit
        CosTheta = ExponentialDecay(ratio, 0.2572123, 14.66644824, 0.74675075)
        
        return CosTheta, GetNearAnodePhis()[peakAnode-6] + phi_err
    else:
        ratio = (amplitudes[anodeFarAntiClockwise] + amplitudes[anodeFarClockwise])/ amplitudes[peakAnode]
        if ratio < 0.08:
            # May data points fall close to zero but have a reasonably high ratio, drop these to zero so results can be more continuous
            return -0.719, GetNearAnodePhis()[peakAnode-6]
        else:    
            return Cubic(ratio, 0.02284734, -0.02741671, 0.18964751, -0.3044784), GetNearAnodePhis()[peakAnode-6]
    
def LocalisationRadius(anodeRisetime):
    
    # This equation was the best fit, found using SciDavies
    return 5.513318 * anodeRisetime *(1/5) + 0.416806

#Path Reconstruction

def PathReconstruction(_paths):
        
    angleError = []
    DistanceError = []
        
    for i in _paths:
        tree = uproot.open(i)
        events = tree['simul;1/InitialParticleType'].array(library = 'np').shape[0]
        
        for j in range(events):
            if(PathEventRejection(tree, j) == True):
                
                # Find true path, returns none if path not straight, else returns cartesian positions of first and last electrons
                TruePath = TruePathFinding(tree, j)
                if (TruePath != None):
                    
                    firstElectron, lastElectron = TruePath
                    
                    # Path reconstruction algorithm based on signals and time delays
                    l = 0
                    anodeAmplitudes = np.zeros(11)
                    anodePeakTimes = np.zeros(11)
                    for k in GetAnodes():
                        anodeAmplitudes[l] = tree[f'simul;1/{k}Amplitude'].array(library = 'np')[j][0]
                        anodePeakTimes[l] = tree[f'simul;1/{k}CurrentPeakTime'].array(library = 'np')[j][0]
                        l = l + 1
                    
                    # Returns two vectors to the path, can plot a line through them, only works if 3 anodes are triggered with correct geometry
                    vecs = PathFinding(anodeAmplitudes, anodePeakTimes - np.min(anodePeakTimes))
                    
                    if (vecs != None):
                        print("yes")
                        #Calculating angular error
                        TrueDirection = Normalize(lastElectron - firstElectron)
                        CalculatedDirection = Normalize(vecs[0] - vecs[1])
                        angleError.append(abs(np.dot(TrueDirection, CalculatedDirection)))
                        
                        #Calculating distance error
                        cross = np.cross(TrueDirection, CalculatedDirection)
                        DistanceError.append(abs(np.dot(cross, (firstElectron - vecs[0]))))
                        
    
    plt.xlabel("Angle between direction vectors (rad)")
    plt.ylabel("Count")
    
    plt.hist(np.arccos(angleError), bins = 35)
    print(np.average(np.arccos(angleError)))
    print(np.std(np.arccos(angleError)))
    
    plt.show()
    plt.xlabel("Shortest distance between the calculated and correct paths (cm)")
    
    plt.ylabel("Count")
    
    plt.hist(DistanceError, bins = 35)
    print(np.average(DistanceError))
    print(np.std(DistanceError))

def PathFinding(_anodeAmplitudes, anodeTimeDeltas):
    #Determine the indexes of the anodes with amplitude
    indexes = []
    for i in range(len(_anodeAmplitudes)):
        if _anodeAmplitudes[i] > 0.002: # Arbitrary float for an anode with a significant signal
            indexes.append(i)
    count = len(indexes)
    if count == 3:
        Vectors = GenerateAnodeVectors()
        distances = []
        for i in range(3):
            distances.append(CalculateError(Vectors[indexes[i-1]], Vectors[indexes[i-2]]))
            # 2-3, 1-3, 2-1
            
        distanceSum = sum(distances)
        if(distanceSum < 3.85 and distanceSum > 3.75):
            # Anodes Align and signal can be used
            # Find Central anode
            centralAnode = np.argmax(distances)
            
            # Central Anode Found,
            centralAnodeVector = PolarToCartesian(Vectors[indexes[centralAnode]])
            anodeVectorOne = PolarToCartesian(Vectors[indexes[centralAnode - 1]])
            anodeVectorTwo = PolarToCartesian(Vectors[indexes[centralAnode - 2]])
            
            timeDifferences = [0, 0, 0]
            for i in range(3):
                timeDifferences[i] = anodeTimeDeltas[indexes[centralAnode - i]]
            
            return ThreeVectorPathFit((centralAnodeVector, anodeVectorOne, anodeVectorTwo), timeDifferences)
        else:
            return
    else:
        return 
    
def TruePathFinding(_tree, _event):
    # finds the true path of a particle by finding the cartesian positions of the first and last electron

    #First electron
    r = _tree['simul;1/InitElecRadius'].array(library = 'np')[_event][0]
    cosTheta  = _tree['simul;1/InitElecCosTheta'].array(library = 'np')[_event][0]
    phi = _tree['simul;1/InitElecPhi'].array(library = 'np')[_event][0]
    
    #Last electron
    _r = _tree['simul;1/InitElecRadius'].array(library = 'np')[_event][-1]
    _cosTheta = _tree['simul;1/InitElecCosTheta'].array(library = 'np')[_event][-1]
    _phi = _tree['simul;1/InitElecPhi'].array(library = 'np')[_event][-1]
    R = PolarToCartesian((r, cosTheta, phi))
    _R = PolarToCartesian((_r, _cosTheta, _phi))
    
    for i in range(len(_tree['simul;1/InitElecRadius'].array(library = 'np')[_event])): # Checks that each electron in the event was created on the line
        # For each electron
        e_r = _tree['simul;1/InitElecRadius'].array(library = 'np')[_event][i]
        e_cosTheta  = _tree['simul;1/InitElecCosTheta'].array(library = 'np')[_event][i]
        e_phi = _tree['simul;1/InitElecPhi'].array(library = 'np')[_event][i]
        
        
        if DistanceTest(R - PolarToCartesian((e_r, e_cosTheta, e_phi)), R - _R):
            return
    
    # Electrons for a line
    return (R, _R)
    
def ThreeVectorPathFit(Vectors, dts):
    
    # Mathematics detailed in the report to convert the three vectors into a plane, and to solve for the correct initial time to fit a straight line
    centralAnodeVector, anodeVectorOne, anodeVectorTwo = Vectors
    

    #Get straight line vector passing cental anode:
    directionVector = anodeVectorTwo - anodeVectorOne
    
    directionVector =  Normalize(directionVector)
    
    #Create vector out of plane with direction vector
    offsetVector = centralAnodeVector - anodeVectorOne
    
    #Offset vector
    offsetVector = offsetVector - directionVector * np.dot(directionVector, offsetVector) # subtract all maginutde of offset vector in line with direction to get the shortest vector from line to central anode
    
    offsetVector = offsetVector * 1/3  # 1/3 length offset minimises the square distance from the line 
    
    # All vectors brought into plane and normalized
    centralAnodeVector = Normalize(centralAnodeVector - 2 * offsetVector) # offset central anode
    anodeVectorOne = Normalize(anodeVectorOne + offsetVector)
    anodeVectorTwo = Normalize(anodeVectorTwo + offsetVector)

    #Sin and cos variables for function
    SinFirst = np.dot(centralAnodeVector, anodeVectorOne)
    SinSecond = np.dot(centralAnodeVector, anodeVectorTwo)
    
    CosFirstCentralAngle = (1-SinFirst*SinFirst)**(1/2)
    CosSecondCentralAngle = (1-SinSecond*SinSecond)**(1/2)
    
    # Numerical Fit using Newton Raphson
    lastValue = 1
    cycles = 0
    accuracy = False
    
    #Central = 2, first = 1, second = 3, from the mathematics used 
    
    while(accuracy == False):
        
        radii = [0, 0, 0]
        
        #Calculate current radii
        for i in range(3):
            radii[i] = radiusCacluation(lastValue + dts[i])
        
        # Put last value of radii into  the function, equation detailed in project report
        f_last = (radii[2] * SinFirst - radii[0]) * radii[1] * CosSecondCentralAngle + (radii[1]* SinSecond - radii[0]) * radii[2] * CosFirstCentralAngle
        
        df_last = GradientCalculation(lastValue + dts[2], lastValue + dts[1])* (CosSecondCentralAngle * SinFirst + SinSecond * CosFirstCentralAngle) - GradientCalculation(lastValue + dts[0], lastValue + dts[1]) * CosSecondCentralAngle - GradientCalculation(lastValue + dts[2], lastValue + dts[0] ) * CosFirstCentralAngle
        
        newValue = lastValue - f_last/df_last
        
        #Check if accuracy found
        cycles = cycles + 1
        if (abs(lastValue -newValue) < 1):
            accuracy = True
            
        lastValue = abs(newValue)
        cycles = cycles + 1
        if cycles > 16 :
            # Value not converging
            return
    
    # lastValue is now best estimate of time
    
    return centralAnodeVector* radiusCacluation(lastValue + dts[0]), anodeVectorOne * radiusCacluation(lastValue + dts[2]) # Return scaled vectors that lie on line

# Finds the drift time of each electron
def FindDriftTime(_paths):
    
    # This function is just grabbing data out of the root file and plotting it.
    
    radii = []
    driftTime = []
    for i in _paths:
        tree = uproot.open(i)
        events = tree['simul;1/InitialParticleType'].array(library = 'np').shape[0]
        
        for j in range(events):
            if(PathEventRejection(tree, j) == True):
                TruePath = TruePathFinding(tree, j)
                if (TruePath != None):
                    
                    for k in tree['simul;1/InitElecRadius'].array(library = 'np')[j]:
                        radii.append(k)
                    for k in tree['simul;1/FinElecTime'].array(library = 'np')[j] - tree['simul;1/InitElecTime'].array(library = 'np')[j]:
                        driftTime.append(k)
    
    
    #Eliminate erroneous values with large radii and small drift times
    rejectionCount = 0
    
    for i in range(len(driftTime)):
        i = i - rejectionCount
        if driftTime[i] < 10000:
            if radii[i] > 10:
                radii.pop(i)
                driftTime.pop(i)
                rejectionCount = rejectionCount + 1
    
    
    plt.scatter(driftTime, radii)
    plt.xlabel("drift time (arbitrary time units)")
    plt.ylabel("initial radius (cm)")
    
    #Export data, commented out for now
    #df = pd.DataFrame({'driftTime': driftTime[1::500], 'radius': radii[1::500]})
    #df.to_excel('driftTimeAginstRadiusConcise.xlsx', sheet_name='sheet1')


# These three functions  are all components used to solve the function for the path reconstruction (gradient for Newton Raphson)
def GradientCalculation(t1, t2):
    return radiusCacluation(t1) * GradientComponentCalculation(t2) + radiusCacluation(t2) * GradientComponentCalculation(t1)
    
def GradientComponentCalculation(t):
    powerFactor = 0
    if t != 0:
       powerFactor = t**-0.12967797
    else:
        t = 1
    return -28.7176693 * np.exp(-0.00003553920 * t) * (-0.12967797 * powerFactor/t -0.00003553920 * powerFactor) 

def radiusCacluation(t):
    rad = 15 - 28.7176693 *t**-0.12967797 * np.exp(-0.00003553920 * t)
    
    if abs(rad) == np.inf:
        return 0
    
    return rad

#----------Management Functions----------#

def EventRejection(_tree, _event):  # Reject localised deposition event if there is no data, or if the electrons are too spread out in r, cosTheta or phi
    statusArray = _tree['simul;1/FinElecStatus'].array(library = 'np')[_event]
    if (statusArray.shape == (0,)):
        #print('Event did not occur properly')
        return False
    elif abs(np.average(_tree['simul;1/InitElecCosTheta'].array(library = 'np')[_event]) - _tree['simul;1/InitElecCosTheta'].array(library = 'np')[_event][0]) > 0.05:
        #Erronious cosTheta
        return False
    elif abs(np.average(_tree['simul;1/InitElecPhi'].array(library = 'np')[_event]) - _tree['simul;1/InitElecPhi'].array(library = 'np')[_event][0]) > 0.05:
        return False
    elif abs(np.average(_tree['simul;1/InitElecRadius'].array(library = 'np')[_event]) - _tree['simul;1/InitElecRadius'].array(library = 'np')[_event][0]) > 0.05:
        return False
    return True

def GetMultipleFilePaths(_mainDirectory, _currentDirectory, _rootFileNameCore, rootFileNameSpecific): # Returns an array of file paths
    RootFiles = []
    for i in rootFileNameSpecific:
        RootFiles.append(f'{_mainDirectory}{_currentDirectory}/{_rootFileNameCore}{i}.root')
    return RootFiles


def ColourGradient(factors): #Used for plotting coloured plots. returns an array of colours based on the input values, going from red to green
    factors = factors/np.max(factors)
    colours = np.zeros((len(factors), 3))
    colours[:, 0] = np.sqrt(factors)
    colours[:, 1] = np.sqrt(1-factors)
    return colours

def RescaleArray(_array): # Scales the array so the max value is one
    minVal = np.min(_array)
    _array =_array - minVal
    return _array/np.max(_array)

def PathEventRejection(_tree, _event): # Rejects events that did not properly occur. Different to the rejection that goes on in the TruePathFinding() function
    statusArray = _tree['simul;1/FinElecStatus'].array(library = 'np')[_event]
    if (statusArray.shape == (0,)):
        return False
    return True

#----------MATHS FUNCTIONS----------#

def CalculateError(V1, V2): # Calculate the distance between two polar positions
    r, cos, phi = V1
    _r, _cos, _phi = V2
    dr = r* r + _r* _r - 2 * r* _r * (cos * _cos + np.cos(phi- _phi) * ((1-cos**2) * (1-_cos**2))**0.5)
    return dr**0.5

def PolarToCartesian(R): # Returns a cartesian Vector for a polar input
    r, cosTheta, phi = R
    sinTheta = (1-cosTheta*cosTheta)**0.5
    return np.array((r * sinTheta * np.cos(phi), r* cosTheta, r * sinTheta * np.sin(phi)))

def DistanceTest(R, D): # Finds the shortest distance between a line and the origin, and checks that it is small, could rewrite this using vector products from numpy if needs be
    distance = (R[0]*R[0]+R[1]*R[1]+R[2]*R[2]-(R[0]*D[0]+R[1]*D[1]+R[2]*D[2])**2/(D[0]*D[0]+D[1]*D[1]+D[2]*D[2]))**0.5
    if distance > 0.3: # Arbitrary distance
        return True
    return False

def Normalize(Vector): # Normalizes vector
    return Vector/np.sqrt(np.dot(Vector, Vector))

#----------VARIABLE GENERATORS----------#

def GenerateAnodeVectors(): # Generate spherical polar coordinates for the anode vectors
    #r, costheta, phi
    anodeDirections = np.zeros((11, 3))
    anodeDirections[0] = (1, 1, 0)
    farAnodePhis = np.linspace(np.pi/10, np.pi/10 - 2* np.pi, 5, endpoint = False)
    nearAnodePhis = np.linspace(3*np.pi/10, 3*np.pi/10 - 2* np.pi, 5, endpoint = False)
    for i in range(0, 5):
        anodeDirections[i+1] = (1, 0.45, farAnodePhis[i])
        anodeDirections[i+6] = (1, -0.45, nearAnodePhis[i])
    return anodeDirections
    
def GenerateAnodeVectorsCartesian():# Generate cartesian coordinates for the anode vectors
    anodeDirections = np.zeros((11, 3))
    anodeDirections[0] = (0, 1, 0)
    farAnodePhis = np.linspace(np.pi/10, np.pi/10 - 2* np.pi, 5, endpoint = False)
    nearAnodePhis = np.linspace(3*np.pi/10, 3*np.pi/10 - 2* np.pi, 5, endpoint = False)
    for i in range(0, 5):
        anodeDirections[i+1] = (0.89 * np.cos(farAnodePhis[i]), 0.45, 0.89 * np.sin(farAnodePhis[i]))
        anodeDirections[i+6] = (0.89 * np.cos(nearAnodePhis[i]), -0.45, 0.89 * np.sin(nearAnodePhis[i]))
    return anodeDirections

def GetColours(): # An array of 9 colours
    return [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0.5, 0), (0, 1, 0.5), (0.5, 0, 1)]

def GetAnodes():  # Returns an array of anode names
    anodes = []
    for i in range(11):
        if(i < 6):
            anodes.append(f'F{i}')
        else:
            anodes.append(f'N{i-5}')
    return anodes

def GetFarAnodePhis():  # Get phi of the far field anodes
    anodePhis = np.linspace(np.pi/10, np.pi/10 - 2* np.pi, 5, endpoint = False)
    
    # Cast last two phis (< -pi) into positive phi 
    anodePhis[3:5] = anodePhis[3:5]+ 2 * np.pi
    return anodePhis

def GetNearAnodePhis():  # Get phi of the far field anodes
    anodePhis = np.linspace(3*np.pi/10, -17*np.pi/10, 5, endpoint = False)
    
    # Cast last two phis (< -pi) into positive phi 
    anodePhis[-1] = anodePhis[-1]+ 2 * np.pi
    return anodePhis

#----------Determining Generic Properties----------#

def HistogramOfElectronsIonised(_paths): # Counts the number of electrons ionised for every event in the paths
    
    electronsCreated = []
    
    for i in _paths:
        tree = uproot.open(i)
        events = tree['simul;1/InitialParticleType'].array(library = 'np').shape[0]
        
        for j in range(events):
            if(EventRejection(tree, j) == True):
                electronsCreated.append(tree['simul;1/NumInitElecs'].array(library = 'np')[j])
        
    electronsCreated = np.array(electronsCreated)
    
    plt.title(f'Histogram counting the number of electrons released for a 5.6GeV muon')
    plt.xlabel('Number of electrons')
    plt.ylabel('Count')
    
    plt.hist(electronsCreated, bins = 70)
    plt.xlim(0, 200)
    
def DetermineParticleRange(_paths): # Calculates the average range of a photon in the simulation
    
    for i in _paths:
        tree = uproot.open(i)
        events = tree['simul;1/InitialParticleType'].array(library = 'np').shape[0]
        
        particleRadius = tree['simul;1/InitialParticleRadius'].array(library = 'np')
        particleCos = tree['simul;1/InitialParticleCosTheta'].array(library = 'np')
        particlePhi = tree['simul;1/InitialParticlePhi'].array(library = 'np')
        
        
        electronRadius = tree['simul;1/InitElecRadius'].array(library = 'np')
        electronCosTheta = tree['simul;1/InitElecCosTheta'].array(library = 'np')
        electronPhi = tree['simul;1/InitElecPhi'].array(library = 'np')
        
        dist = []
        
        for j in range(events):
            if(EventRejection(tree, j) == True):
                
                
                vector = CalculateError((particleRadius[j], particleCos[j], particlePhi[j]), (electronRadius[j][0], electronCosTheta[j][0], electronPhi[j][0]))
                dist.append(vector)
                
    dist = np.array(dist)
    
    print(np.average(dist))
    
def CountFails(_paths):  # Used to investigate the number of evenets that failed and why
    d = 0  # Event did not occur
    c = 0  # Initial electrons wer broadly spread in CosTheta
    p = 0  # Initial electrons wer broadly spread in phi
    
    for i in _paths:
        tree = uproot.open(i)
        events = tree['simul;1/InitialParticleType'].array(library = 'np').shape[0]
        
        for j in range(events):
            if(tree['simul;1/FinElecStatus'].array(library = 'np')[j].shape == (0,)):
                d = d + 1
            elif abs(np.average(tree['simul;1/InitElecCosTheta'].array(library = 'np')[j]) - tree['simul;1/InitElecCosTheta'].array(library = 'np')[j][0]) > 0.1:
                c = c + 1
                
            elif abs(np.average(tree['simul;1/InitElecPhi'].array(library = 'np')[j]) - tree['simul;1/InitElecPhi'].array(library = 'np')[j][0]) > 0.1:
                p = p + 1
    print(d)
    print(c)
    print(p)

#----------Fitting Functions----------#

def Cubic(x, A, B, C, D):
    return A * x**3 + B * x**2 + C * x +  D

def Quartic(x, A, B, C, D, E):
    return A * x**4 + B * x**3 + C * x**2 + D * x + E

#Equation used for path reconstruction radius radius from drift time
def RadiusEquation(x, factor, power, decay):
    rad = 15 + factor *x**power * np.exp(decay * x)
    return rad

# Exponential decay from one value to another
def ExponentialDecay(x, A, k, C):
    return A * (1-np.exp(-k * x)) - C

#----------PLOTTING FUNCTIONS USING INITIAL ELECTRON LOCATIONS----------#

def InitialElectronPosPolar(Tree, particleIndex):
    #Access Data
    #Tree = uproot.open(_path)
    ElecCos = Tree['simul;1/InitElecCosTheta'].array(library = 'np')
    ElecPhi = Tree['simul;1/InitElecPhi'].array(library = 'np')
    
    plt.scatter(ElecPhi[particleIndex], ElecCos[particleIndex])
                
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-1, 1)
    
    plt.show()

def InitialElectronCosThetaRad(_path, particleIndex):
    #Access Data
    Tree = uproot.open(_path)
    ElecCos = Tree['simul;1/InitElecCosTheta'].array(library = 'np')
    ElecRad = Tree['simul;1/InitElecRadius'].array(library = 'np')
    
    plt.scatter(ElecRad[particleIndex], ElecCos[particleIndex])
                
    plt.xlim(0, 15)
    plt.ylim(-1, 1)
    
    plt.show()

def InitialElectronsPos(Tree, particleIndex = 0):
    
    Tree = uproot.open(Tree)
    
    #Access Data
    ElecRad = Tree['simul;1/InitElecRadius'].array(library = 'np')[particleIndex]
    ElecCos = Tree['simul;1/InitElecCosTheta'].array(library = 'np')[particleIndex]
    ElecPhi = Tree['simul;1/InitElecPhi'].array(library = 'np')[particleIndex]
    
    t = ElecPhi[0] +np.pi/2
    t = 0
    deg = np.linspace(0, np.pi *2)
    r = 15
    
    plt.scatter(ElecRad * np.sqrt(1-np.square(ElecCos)) * np.cos(ElecPhi + t), ElecRad * ElecCos, s = 1)
    
    #plot shell of detector
    plt.plot(r * np.cos(deg), r * np.sin(deg))
    plt.show()
    return t

def FinalElectronsPos(_path, particleIndex = 0):
    #Access Data
    Tree = uproot.open(_path)
    ElecRad = Tree['simul;1/FinElecRadius'].array(library = 'np')
    ElecCos = Tree['simul;1/FinElecCosTheta'].array(library = 'np')
    ElecPhi = Tree['simul;1/FinElecPhi'].array(library = 'np')
    
    t = 0
    
    deg = np.linspace(0, np.pi *2)
    r = 15
    
    plt.scatter(ElecRad[particleIndex] * np.sqrt(1-np.square(ElecCos[particleIndex])) * np.cos(ElecPhi[particleIndex] + t), ElecRad[particleIndex] * ElecCos[particleIndex], s = 1)
    
    #plot shell of detector
    plt.plot(r * np.cos(deg), r * np.sin(deg))
    plt.show()

def FinalElectronPosPolar(_path, particleIndex):
    #Access Data
    Tree = uproot.open(_path)
    ElecCos = Tree['simul;1/FinElecCosTheta'].array(library = 'np')
    ElecPhi = Tree['simul;1/FinElecPhi'].array(library = 'np')
    
    plt.scatter(ElecPhi[particleIndex], ElecCos[particleIndex])
                
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-1, 1)
    
    plt.show()

#----------PROCESSING LAB DATA----------#

#this method processes the data read in from the file into a useful format, copied from lab team
def processDataFrame(df):
    df['amplitude'] = [pd.to_numeric(x.replace('[','').replace(']','').replace('\n','').replace('  ',' ').split(' ')) for x in df['amplitude']]
    df['baseline_std'] = [pd.to_numeric(x.replace('[','').replace(']','').replace('\n','').replace('  ',' ').split(' ')) for x in df['baseline_std']]
    df['peak'] = [pd.to_numeric(x.replace('[','').replace(']','').replace('\n','').replace('  ',' ').split(' ')) for x in df['peak']]
    df['risetime'] = [pd.to_numeric(x.replace('[','').replace(']','').replace('\n','').replace('  ',' ').split(' ')) for x in df['risetime']]
    df['width'] = [pd.to_numeric(x.replace('[','').replace(']','').replace('\n','').replace('  ',' ').split(' ')) for x in df['width']]
    
    df['amplitude'] = [ x[ ~np.isnan(x)]*1000 for x in df['amplitude']] #converts from V to mV
    df['baseline_std'] = [ x[ ~np.isnan(x)]*1000 for x in df['baseline_std']] #converts from V to mV
    df['peak'] = [ x[ ~np.isnan(x)] for x in df['peak']]
    df['risetime'] = [ x[ ~np.isnan(x)] for x in df['risetime']]
    df['width'] = [ x[ ~np.isnan(x)] for x in df['width']]
    return df

def TestEventLocalisation(fileName): # Testing the events of the lab team's data
    #read in the data
    df1 = pd.read_csv(f"{fileName}.csv")
    
    #process the data and save it in this dataframe
    df1 = processDataFrame(df1)
    
    # Get number of events
    events = len(df1['amplitude'])
    t = np.pi/2
    
    X = []
    Y = []
    
    CalibrationArray = np.array([0.89407,0.43494,0.86838,0.59066,0.43082,0.35467,0.71839,0.78081,0.96399,0.47646,0.58817])
    
    for r in np.linspace(100, 1000, 100):
        for i in range(events):
            eventPosition = LocalisationAlgorithm(np.divide(df1['amplitude'][i], CalibrationArray), df1['risetime'][i])
            if eventPosition != None:
                X.append(eventPosition[0]/r * (1 - eventPosition[1] * eventPosition[1])**0.5 * np.cos(eventPosition[2] + t))
                Y.append(eventPosition[0]/r * eventPosition[1])
        
        plt.scatter(X, Y)
        v = np.linspace(0, 2 * np.pi)
        plt.plot(15* np.cos(v), 15* np.sin(v))
        plt.show()
        X.clear()
        Y.clear()

if __name__ == '__main__':
    main()