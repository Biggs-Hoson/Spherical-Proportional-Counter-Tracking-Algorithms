# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:10:54 2023

@author: William Lewin

This program attempts to find the path of a relativistic, weakly ionising particle in the 3d detector using the simulation data,
More code will be needed to be added to this program to get it to work with the real data

"""

#imports
import numpy as np
import matplotlib.pyplot as plt
import uproot
from matplotlib.animation import FuncAnimation

# Vector 3 Class for spherical polar coordinates
class Vector3D:
        if(cartesian):
            # _r = x, _cosTheta = y, _phi = z
            self.radius = np.sqrt(_r*_r + _cosTheta * _cosTheta + _phi * _phi)
            self.cosTheta = np.cos(np.arctan(np.sqrt(_r*_r + _cosTheta * _cosTheta)/_phi))
            self.phi = np.arctan(_cosTheta/_r)
        else:
            self.radius = _r
            self.cosTheta = _cosTheta
            self.phi = _phi
        
    def __sub__(self, other):
        cartesianResults = self.cartesian() - other.cartesian()
        
        _r = np.linalg.norm(cartesianResults)
        _cosTheta = cartesianResults[1]/_r
        _phi = np.arctan(cartesianResults[2]/cartesianResults[0])
        return Vector3D(_r, _cosTheta, _phi)
    
    def __add__(self, other):
        cartesianResults = self.cartesian() + other.cartesian()
        
        _r = np.linalg.norm(cartesianResults)
        _cosTheta = cartesianResults[1]/_r
        _phi = 0
        if(cartesianResults[0] == 0):
            _phi = np.pi/2 * np.sign(cartesianResults[2])
        else:
            _phi = np.sign(cartesianResults[2]) * np.arccos(cartesianResults[0]/(cartesianResults[0]**2 + cartesianResults[2]**2)**0.5)
        return Vector3D(_r, _cosTheta, _phi)
    
    def cartesian(self):
        sinT = np.sqrt(1-self.cosTheta*self.cosTheta)
        x = self.radius * sinT* np.cos(self.phi)
        y = self.radius * self.cosTheta
        z = self.radius * sinT* np.sin(self.phi)
        return np.array([x,y,z])
    
    def prt(self):
        print(self.radius, self.cosTheta, self.phi)
    
    def prtCartesian(self):
        print(self.cartesian())
        
    def PlotPos(self, phase = 0):
        cartesian = self.cartesian()
        X = cartesian[0]*np.cos(phase) + cartesian[2] * np.sin(phase)
        Y = cartesian[1]
        return X, Y
            
def main():
    plt.style.use('dark_background')
    
    #Get Root Files
    mainDirectory = 'C:/Users/willi/Documents/Uni Work/Year 3/Group Project'
    currentSubDirectory = ''
    
    # Root file names are stored as a main section and a number
    fileNameCore = 'PathReconstruction1'
    fileNumber = ['0', '1', '2']
    
    paths = GetMultipleFilePaths(mainDirectory, currentSubDirectory, fileNameCore, fileNumber)
    firstPath = paths[0]
    
    
    path = 'C:/Users/willi/Documents/Uni Work/Year 3/Group Project/GammaData/output_0.root'
    
    print(uproot.open(path)[f'simul;1/InitialParticleEnergy'].array(library = 'np')[0])
    
    plt.title('A 2 GeV Muon created at (r = 14.9, cosTheta = 0.866, phi = 0), travelling downwards')
    plt.ylabel('y pos (cm)')
    plt.xlabel('x pos (cm)')
    InitialElectronsPos(path, 1)
    
    #for i in np.linspace(0, 2 * np.pi, 10):
    #    MultiplePathViewer(firstPath, [], i)
        
    
    #print(firstPath)
    
    #Tree = uproot.open(firstPath)[f'simul;1/NumInitElecs'].array(library = 'np')
    #print(Tree)
    
    
    
    #DirectionMethodTwo(firstPath)
    
    
#----------Investigations----------#

def DirectionMethodTwo(_path):
    Tree = uproot.open(_path)
    total_Events = numParticles(Tree)
    
    print(total_Events)
    
    events = range(total_Events)
    
    amplitudes = []
    for i in GetAnodes():
        amplitudes.append(Tree[f'simul;1/{i}Amplitude'].array(library = 'np'))
    
    amplitudes = np.array(amplitudes).transpose()
    
    truePhi = []
    ratio = []
    
    farAnodePhis = np.linspace(np.pi/10, np.pi/10 - 2* np.pi, 5, endpoint = False)
    
    for i in events:
        
        statusArray = Tree['simul;1/FinElecStatus'].array(library = 'np')[i]
        if (statusArray.shape != (0,)):
            if(statusArray[0] == 1):
                #Event triggered successfully
                
                amp = 0
                for j in GetAnodes():
                    if amp == 0:
                        amp = amp + Tree[f'simul;1/{j}Amplitude'].array(library = 'np')[i][0] 
                if amp != 0:
                    # Event created a Signal
                    peakAnode = amplitudes[i].argmax()
                    
                    if peakAnode == 0:
                        # F0
                        
                        #print(i)
                        
                        farFieldAnodes = amplitudes[i, 1:6]
                        
                        # Determine if there are more than 2 significant signals
                        significance = 0.004
                        if np.max(farFieldAnodes) > significance:
                            # There is a far field anode with a significant signal to be used
                            
                            indexMax = np.argmax(farFieldAnodes)
                            
                            # Now that the maximum signal has been found, the signals on the two either side are to be detected
                            leftAnodeSignificant = False
                            rightAnodeSignificant = False
                            
                            indexRight = indexMax + 1
                            if (indexRight == 5):
                                indexRight = 0
                                # Make sure index loops back to 0
                            
                            if farFieldAnodes[indexMax - 1] > significance:
                                leftAnodeSignificant = True
                            if farFieldAnodes[indexRight] > significance:
                                rightAnodeSignificant = True
                            
                            if leftAnodeSignificant == False and rightAnodeSignificant == False:
                                #print('F0: no signifiant third amplitude next to the second')
                                j = 1
                            elif leftAnodeSignificant == True:
                                
                                
                                if rightAnodeSignificant == True:
                                    #Both significant
                                    anodeRatio = farFieldAnodes[indexMax-1]/farFieldAnodes[indexRight]
                                    
                                    print('both')
                                    
                                    if anodeRatio < 0.5:
                                        #right anode twice that of left anode
                                        print('right')
                                        
                                        thisRatio = farFieldAnodes[indexRight]/farFieldAnodes[indexMax]
                                        
                                        ratio.append(thisRatio)
                                        phiDelta = farAnodePhis[indexMax] - Tree['simul;1/InitialParticlePhi'].array(library = 'np')[i][0]
                                        
                                        truePhi.append(phiDelta)
                                        
                                    elif anodeRatio > 2:
                                        #left anode twice that of right
                                        print('left')
                                        
                                        print(i)
                                        
                                        print('main Amplitude')
                                        print(farAnodePhis[indexMax])
                                        print(farFieldAnodes[indexMax])
                                        print('right Amplitude')
                                        print(farAnodePhis[indexRight])
                                        print(farFieldAnodes[indexRight])
                                        print('left')
                                        print(farAnodePhis[indexMax-1])
                                        print(farFieldAnodes[indexMax-1])
                                        
                                        thisRatio = farFieldAnodes[indexMax-1]/farFieldAnodes[indexMax]
                                        
                                        ratio.append(thisRatio)
                                        
                                        realPhi = Tree['simul;1/InitElecPhi'].array(library = 'np')[i][0]
                                        
                                        print(realPhi)
                                        
                                        if realPhi > np.pi/10:
                                            realPhi = realPhi - 2*np.pi
                                            print('used')
                                        
                                        print(realPhi)
                                        
                                        phiDelta = realPhi - farAnodePhis[indexMax]
                                        
                                        print(farAnodePhis[indexMax])
                                        print(phiDelta)
                                        
                                        truePhi.append(phiDelta)
                                        
                                    else:
                                        i = 1
                                        #Somewhat ballenced
                                    
                                    
                                else:
                                    j = 0
                                    
                                    
                                    #Left anode significant only                                
                                    
                                
                            else:
                                i = 0
                                #Right anode Only
                                   
                                
                        else:
                            #print('F0: No significant second amplitude')
                            j = 1
                        
                        
                        
                        
                          
                        
                    #elif peakAnode > 5:
                        # Near anodse
                        #print('near')
                    #else:
                        # F1 through 5
                        #print('far')
    plt.scatter(truePhi, ratio)
    plt.show                    
                        
        

def DirectionErrorInvestigation(_path):
    #This method proved futile as the greatest strength anode was the only one with adequate signal for individual electrons
    
    Tree = uproot.open(_path) 
    
    # Find Number of particles
    events = numParticles(Tree)
    
    cosThetas = []
    cosDeviation = []
    
    failCount = 0
    
    
    for i in range(22, 23):
        #Skip events if they produced no amplitude or arent valid
        if (eventIsValid(Tree, i) == True):
            print(i)
            calculatedDirection = SignalDirection(Tree, i)
            #Get actual electron direction
            
            #realpos = Vector3D()
            
            #realpos.radius = Tree['simul;1/InitialParticleRadius'].array(library = 'np')[i][0]
            #realpos.cosTheta = Tree['simul;1/InitialParticleCosTheta'].array(library = 'np')[i][0]
            #realpos.phi = Tree['simul;1/InitialParticlePhi'].array(library = 'np')[i][0]
            
            cosThetaActual = Tree['simul;1/InitialParticleCosTheta'].array(library = 'np')[i][0]
            cosThetaCalculated = calculatedDirection.cosTheta
            
            
            err = cosThetaCalculated -cosThetaActual #cosThetas[i - failCount]
            print(cosThetaActual)
            print(cosThetaCalculated)
            print(err)
            
            
            cosThetas.append(cosThetaActual)
            cosDeviation.append(cosThetaCalculated)
        else:
            failCount = failCount + 1
    
    
    print(cosThetas)
    print(cosDeviation)
    
    plt.scatter(cosThetas, cosDeviation)
    plt.show
    
#----------Management Functions----------#

def GetMultipleFilePaths(_mainDirectory, _currentDirectory, _rootFileNameCore, rootFileNameSpecific):
    RootFiles = []
    for i in rootFileNameSpecific:
        RootFiles.append(f'{_mainDirectory}{_currentDirectory}/{_rootFileNameCore}{i}.root')
    return RootFiles

def GetAnodes():
    # This would have been faster to type out, but this looks more clean
    anodes = []
    for i in range(11):
        if(i < 6):
            anodes.append(f'F{i}')
        else:
            anodes.append(f'N{i-5}')
    return anodes

def GetAnodes3D():
    anodePos = []
    anodePos.append(Vector3D(1, 1, 0))
    for i in range(5):
        anodePos.append(Vector3D(1, 0.45, np.pi/10 - i * 2* np.pi /5))
    for i in range(5):
        anodePos.append(Vector3D(1, -0.45, 3* np.pi/10 - i * 2* np.pi /5))
    return anodePos

def numParticles(_Tree):
    return np.shape(_Tree['simul;1/InitElecRadius'].array(library = 'np'))[0]
    
def eventIsValid(_tree, _event):
    statusArray =_tree['simul;1/FinElecStatus'].array(library = 'np')[_event]
    if (statusArray.shape == (1,)):
        if(statusArray[0] != 1):
            return False
    else:
        return False
    amp = 0
    for i in GetAnodes():
        amp = amp + _tree[f'simul;1/{i}Amplitude'].array(library = 'np')[_event][0]
        if(amp != 0):
            return True
    return False


#----------TRUE PATH FUNCTIONS----------#
    
# Function to plot the paths based on the first and last electron created
def plotTruePath(_path):
    
    Tree = uproot.open(_path) 
    
    # Find Number of particles
    particles = np.shape(Tree['simul;1/InitElecRadius'].array(library = 'np'))[0]
    
    for i in range(particles):
        truePath(Tree, i)
    
    #Plot shell of detector
    deg = np.linspace(0, np.pi *2)
    r = 15
    plt.plot(r * np.cos(deg), r * np.sin(deg))
    plt.show

# This function calculates the true path of the particle assuming it to be a straight line
def truePath(_tree, _particleIndex, _doNotPlot = False):
    # Should ideally use a smallest volume fit of all the electron inital locations
    # This option has been chosen for brevity
    
    #Get position of all electron
    ElecRad = _tree['simul;1/InitElecRadius'].array(library = 'np')[_particleIndex]
    ElecCos = _tree['simul;1/InitElecCosTheta'].array(library = 'np')[_particleIndex]
    ElecPhi = _tree['simul;1/InitElecPhi'].array(library = 'np')[_particleIndex]
    
    # Get position of the first and last electrons
    particleInitialPos = Vector3D(ElecRad[0], ElecCos[0], ElecPhi[0])
    particleFinalPos = Vector3D(ElecRad[-1], ElecCos[-1], ElecPhi[-1])
  
    # Draw a line between the first and last electrons created
    if(_doNotPlot == False):
        PlotPath(particleInitialPos, particleFinalPos)
    
    #Return start and end of path
    return particleInitialPos, particleFinalPos

# Plots an individual path based on the start and end locations
def PlotPath(_start, _end, _phase = 0):
    startCartesian = _start.cartesian()
    endCartesian = _end.cartesian()
    
    Y = [startCartesian[1], endCartesian[1]]
    X = [startCartesian[0] * np.cos(_phase) + startCartesian[2] * np.sin(_phase), endCartesian[0] * np.cos(_phase) + endCartesian[2] * np.sin(_phase)]
    
    plt.plot(X, Y)


def PlotAchinosSignals(_VectorArray, actual):
    phase = 0
    for i in _VectorArray:
        X, Y = i.PlotPos(phase)
        plt.plot([0, X], [0, Y])
    
    actual.radius = 0.1
    actual.prt()
    real = actual.PlotPos(phase)
    plt.scatter([real[0]], [real[1]])
    plt.show
    
#----------PLOTTING FUNCTIONS USING INITIAL ELECTRON LOCATIONS----------#

#Simple file to open a root file and plot a single property property
def RootViewer(_path, particleIndex = 0):
    Tree = uproot.open(_path)
    
    #Get signal from simulation
    current = Tree['simul;1/Signal'].array(library = 'np')
    
    #Set a start index
    start = 6200
    
    # Used to plot only 1/m points if needed
    m = 1
    
    # Plot for all particles
    for i in range (current.shape[0]):        
        plotData = current[i]
    
        plotPoints = plotData[start:6400:m]
    
        plt.plot(np.arange(plotPoints.size) +start, -plotPoints)
        
        
    plt.title("Signal from Simulated Detector")
    plt.ylabel("Signal")
    plt.xlabel("Arbitrary Time Scale")
    
    plt.show()

# Animated plot of the locations of ionisation for all particles in a root file
def AnimatedPathViewer():
    
    #Access Data
    Tree = uproot.open('C:/Users/willi/Documents/Uni Work/Year 3/Group Project/output.root')
    ElecRad = Tree['simul;1/InitElecRadius'].array(library = 'np')
    ElecCos = Tree['simul;1/InitElecCosTheta'].array(library = 'np')
    ElecPhi = Tree['simul;1/InitElecPhi'].array(library = 'np')
    
    _frames = 400
    
    
    angle = 2*np.pi/_frames
    
    # Artists
    numArtists = 5
    
    fig, ax1 = plt.subplots()
    artists = []
    
    X = []
    Y = []
    T = []
    
    colours = ['r', 'b', 'g', 'y', 'w']
    
    for i in range(numArtists):
        artistInstance, = ax1.plot([], [], colours[i%len(colours)])
        artists.append(artistInstance)
        X.append(ElecRad[i] * np.sqrt(1-np.square(ElecCos[i])))
        Y.append(ElecRad[i] * ElecCos[i])
        T.append(ElecPhi[i])
    
    def init():
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(9, 15)
        
        ax1.set_title("Paths of Protons in 3D space")
        ax1.set_ylabel("Y/cm")
        ax1.set_xlabel("X/cm")
        return artists,
    
    #How the plot changes based on input data
    def animate(t):
        for i in range(numArtists):
            artists[i].set_data(X[i] * np.sin(T[i] + t * angle), Y[i])
        
        return artists

    #plt.plot(x_a * np.sin(), y_a)

    anim = FuncAnimation(fig, animate, init_func=init, frames = _frames, interval = 1)

    return anim

# Plot the locations of ionisation for a single particles in a root file
def InitialElectronsPos(_path, particleIndex = 0):
    
    #Access Data
    Tree = uproot.open(_path)
    ElecRad = Tree['simul;1/InitElecRadius'].array(library = 'np')[particleIndex]
    ElecCos = Tree['simul;1/InitElecCosTheta'].array(library = 'np')[particleIndex]
    ElecPhi = Tree['simul;1/InitElecPhi'].array(library = 'np')[particleIndex]
    
    t = ElecPhi[0] +np.pi/2
    
    deg = np.linspace(0, np.pi *2)
    r = 15
    
    plt.scatter(ElecRad * np.sqrt(1-np.square(ElecCos)) * np.sin(ElecPhi + t), ElecRad * ElecCos, s = 1)
    
    #plot shell of detector
    plt.plot(r * np.cos(deg), r * np.sin(deg))
    plt.show()


def FinalElectronsPos(_path, particleIndex = 0):

    #Access Data
    Tree = uproot.open(_path)
    ElecRad = Tree['simul;1/FinElecRadius'].array(library = 'np')
    ElecCos = Tree['simul;1/FinElecCosTheta'].array(library = 'np')
    ElecPhi = Tree['simul;1/FinElecPhi'].array(library = 'np')
    
    t = ElecPhi[0][0]
    
    deg = np.linspace(0, np.pi *2)
    r = 15
    
    plt.scatter(ElecRad[particleIndex] * np.sqrt(1-np.square(ElecCos[particleIndex])) * np.sin(ElecPhi[particleIndex] + t), ElecRad[particleIndex] * ElecCos[particleIndex], s = 1)
    
    #plot shell of detector
    plt.plot(r * np.cos(deg), r * np.sin(deg))
    plt.show()

def InitialElectronPosPolar(_path, particleIndex):
    #Access Data
    Tree = uproot.open(_path)
    ElecCos = Tree['simul;1/InitElecCosTheta'].array(library = 'np')
    ElecPhi = Tree['simul;1/InitElecPhi'].array(library = 'np')
    
    plt.scatter(ElecPhi[particleIndex], ElecCos[particleIndex])
                
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-1, 1)
    
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

def MultiplePathViewer(_path, indexRange = [], t = 1.7):
    #Access Data
    Tree = uproot.open(_path)
    ElecRad = Tree['simul;1/InitElecRadius'].array(library = 'np')
    ElecCos = Tree['simul;1/InitElecCosTheta'].array(library = 'np')
    ElecPhi = Tree['simul;1/InitElecPhi'].array(library = 'np')
    
    if(len(indexRange) == 0):
        indexRange =  range(ElecRad.shape[0])
        print()
    
    deg = np.linspace(0, np.pi *2)
    r = 15
    
    for i in indexRange:
        if(len(ElecRad[i]) != 0):
            plt.scatter(ElecRad[i] * np.sqrt(1-np.square(ElecCos[i])) * np.sin(ElecPhi[i] + t), ElecRad[i] * ElecCos[i], s = 1)
    
    #plot shell of detector
    plt.plot(r * np.cos(deg), r * np.sin(deg))
    plt.show()

if __name__ == '__main__':
    main()