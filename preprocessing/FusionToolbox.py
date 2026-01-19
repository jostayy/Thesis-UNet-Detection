"""
Kod bazowy udostępniony przez promotora.
Wykorzystany do wstępnego przetwarzania sygnałów wibrotermograficznych.
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:34:19 2024

@author: user
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
# import FileProcessor as FP
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from matplotlib import cm


class FusionToolbox:
    def ReduceResolution(Image,ResolutionReductionFactor):
        Factor = ResolutionReductionFactor
        ReducedImage = Image[::Factor,::Factor]
        # InterpolatedImage =cv2.resize(ReducedImage,None,fx = Factor, fy = Factor,interpolation = cv2.INTER_LINEAR)
        InterpolatedImage =cv2.resize(ReducedImage,None,fx = Factor, fy = Factor,interpolation = cv2.INTER_NEAREST)
        return InterpolatedImage
    

    def Return2ndStates(Damage,Severity):
        DmgStates = [[3,14,15,16],
                      [5,3,4,5],
                      [7,5,6,7],
                      [9,9,10,11],
                      [11,12,13,14],
                      [13,1,2,3],
                      [15,7,8,9],]
        
        for entry in DmgStates:
            if entry[0] == Damage:
                if Severity == 1:
                    return [entry[1], entry[2]]  # Return 2nd and 3rd values
                elif Severity == 2:
                    return [entry[1], entry[3]]  # Return 2nd and 4th values

        return None
        
              
      # D3:  14 ->  15(25%) -> 16(55%)
      # Sensors: 5,6,7,8, 1, 9 
        
      #   # D5:  3  ->  4 (25%) -> 5 (75%)
      # Sensors: 5,6,7,8, 3, 11   
        
      #   # D7:  5  ->  6 (25%) -> 7 (75%)
      # Sensors: 5,6,7,8, 2, 10   
        
      #   # D9:  9  ->  10(25%) -> 11(75%)    # Niedomierzone :(
      # Sensors: 5,6,7,8,     
        
      #   # D11: 12 ->  13(25%) -> 14(75%)
      # Sensors: 5,6,7,8, 4, 12   
        
      #   # D13: 1  ->  2 (50%) -> 3 (75%)
      # Sensors: 5,6,7,8, 3, 11    
        
      #   # D15: 7  ->  8 (25%) -> 9 (75%)
      # Sensors: 5,6,7,8, 2, 10
    
    def PosS(Sensor):  # This works only for one experiment unfortunately :( 
        SensorPosLT = [[1,73,83],
                       [2,71,59],
                       [3,72,34],
                       [4,73,9],
                       [5,41,84],
                       [6,40,60],
                       [7,41,34],
                       [8,41,9],
                       [9,10,85],
                       [10,9,60],
                       [11,10,35],
                       [12,9,9],]
        
        return [SensorPosLT[Sensor-1][1],SensorPosLT[Sensor-1][2]]
    
    def PosSexp2(Sensor):  # 'This needs to be updated!'
        SensorPosLT = [[1,86,12],  
                       [2,61,12],
                       [3,37,12],
                       [4,12,12],
                       [5,86,43],
                       [6,61,43],
                       [7,37,43],
                       [8,12,43],
                       [9,86,74],
                       [10,61,74],
                       [11,37,74],
                       [12,12,74],]
        
        return [SensorPosLT[Sensor-1][1],SensorPosLT[Sensor-1][2]]
    
    
    def PosSensor(Experiment,Sensor):
        if(Experiment == 1):
            SensorPosLT = [[1,73,83],
                           [2,71,59],
                           [3,72,34],
                           [4,73,9],
                           [5,41,84],
                           [6,40,60],
                           [7,41,34],
                           [8,41,9],
                           [9,10,85],
                           [10,9,60],
                           [11,10,35],
                           [12,9,9],]
            
        elif(Experiment == 2):
            SensorPosLT = [[1,86,12],  
                           [2,61,12],
                           [3,37,12],
                           [4,12,12],
                           [5,86,43],
                           [6,61,43],
                           [7,37,43],
                           [8,12,43],
                           [9,86,74],
                           [10,61,74],
                           [11,37,74],
                           [12,12,74],] # This is a "failed attempt at 3mm"
        else:
            print('Invalid experimental number!')
            return None
            
        for entry in SensorPosLT:
            if entry[0] == Sensor:
                return entry[1:]  # Return x, y as a list

        return None
        
    
    
    
    def PosDmg(Experiment,DamageNo):
        
        if(Experiment == 1):
            DmgPos = [[3,60,10],
                      [5,60,36],
                      [7,60,60],
                      [9,60,85],
                      [11,25,10],
                      [13,25,36],
                      [15,25,60],
                      [17,25,85],] # This is a "failed attempt at 3mm"
            
        elif(Experiment == 2):
            DmgPos = [[3,86,58],
                      [5,37,27],
                      [7,61,27],
                      [9,86,27],
                      [11,12,58],
                      [13,37,58],
                      [15,61,58],
                      [4,12,27],] # This is a "failed attempt at 3mm"
        else:
            print('Invalid experimental number!')
            return None
            
        for entry in DmgPos:
            if entry[0] == DamageNo:
                return entry[1:]  # Return x, y as a list

        return None    

    
    
    def CoordinatesFromROI(ROI):
        x1 = ROI[0][0]
        x2 = ROI[0][1]
        y1 = ROI[1][0]
        y2 = ROI[1][1]
        Width = ROI[0][1]-ROI[0][0]
        Height = ROI[1][1]-ROI[1][0]
        Numberx = ROI[0][2]
        Numbery = ROI[1][2]
        spacingx = Width / (Numberx - 1)
        spacingy = Height / (Numbery - 1)
    
        Size = 2*Numberx + 2*Numbery - 4
        Size = int(Size)
        # print('The size is:')
        # print(Size)
        CoordinatesSet = [[x1,y1]]
        for p in np.arange(x1 + spacingx, x2 + spacingx, spacingx):
            CoordinatesSet = np.append(CoordinatesSet, [[p, y1]], axis=0)
        # Vertical sensors
        for l in np.arange(y1 + spacingy, y2, spacingy):
            CoordinatesSet = np.append(CoordinatesSet, [[x1, l]], axis=0)
            CoordinatesSet = np.append(CoordinatesSet, [[x2, l]], axis=0)
        # Last row of sensors
        for q in np.arange(x1, x2 + spacingx, spacingx):
            CoordinatesSet = np.append(CoordinatesSet, [[q, y2]], axis=0)
        CoordinatesSet = np.round(CoordinatesSet).astype(int)
        
        LineWidth = round(max(spacingx,spacingy)/2)
        
        return CoordinatesSet, LineWidth
    
    def CoordinatesWithinROI(ROI):
        x1, x2, Numberx = ROI[0]
        y1, y2, Numbery = ROI[1]
    
        total_points = Numberx * Numbery
    
        x_coords = np.random.uniform(low=x1, high=x2, size=total_points)
        y_coords = np.random.uniform(low=y1, high=y2, size=total_points)
    
        CoordinatesSet = np.column_stack((x_coords, y_coords))
        CoordinatesSet = np.round(CoordinatesSet).astype(int)
    
        # Estimate average spacing for LineWidth
        avg_spacing_x = (x2 - x1) / max(Numberx - 1, 1)
        avg_spacing_y = (y2 - y1) / max(Numbery - 1, 1)
        LineWidth = round(max(avg_spacing_x, avg_spacing_y) / 2)

        return CoordinatesSet, LineWidth
    
    def CoordinatesInGridROI(ROI):
        x1, x2, Numberx = ROI[0]
        y1, y2, Numbery = ROI[1]
    
        Width = x2 - x1
        Height = y2 - y1
        spacingx = Width / (Numberx - 1)
        spacingy = Height / (Numbery - 1)
    
        # Create grid of x and y coordinates
        x_coords = np.linspace(x1, x2, Numberx)
        y_coords = np.linspace(y1, y2, Numbery)
    
        # Meshgrid for full coordinate set
        X, Y = np.meshgrid(x_coords, y_coords)
        CoordinatesSet = np.column_stack((X.ravel(), Y.ravel()))
        CoordinatesSet = np.round(CoordinatesSet).astype(int)
    
        LineWidth = round(max(spacingx, spacingy) / 2)
    
        return CoordinatesSet, LineWidth
    def CoordinatesEdgeBiasedROI(ROI):
        """
        Generate random integer (x, y) coordinates within the ROI, with a higher likelihood 
        of being near the edges rather than in the center.
    
        Parameters:
            ROI (list): [[x_start, x_end, num_x], [y_start, y_end, num_y]]
    
        Returns:
            CoordinatesSet (ndarray): Random coordinates within the ROI, biased toward edges.
            LineWidth (int): Suggested line width based on estimated average spacing.
        """
        x1, x2, Numberx = ROI[0]
        y1, y2, Numbery = ROI[1]
    
        total_points = Numberx * Numbery
    
        def edge_bias_sampler(n, low, high):
            """Sample from a distribution biased toward the edges."""
            u = np.random.rand(n)
            biased = 0.5 - np.abs(u - 0.5)  # Closer to 0.5 → center → gets lower weight
            # Rescale to [0, 1], then to [low, high]
            biased = biased / biased.max()
            return low + (high - low) * biased
    
        x_coords = edge_bias_sampler(total_points, x1, x2)
        y_coords = edge_bias_sampler(total_points, y1, y2)
    
        CoordinatesSet = np.column_stack((x_coords, y_coords))
        CoordinatesSet = np.round(CoordinatesSet).astype(int)
    
        avg_spacing_x = (x2 - x1) / max(Numberx - 1, 1)
        avg_spacing_y = (y2 - y1) / max(Numbery - 1, 1)
        LineWidth = round(max(avg_spacing_x, avg_spacing_y) / 2)
    
        return CoordinatesSet, LineWidth
    def BorderReductionSingleSensor(Image,ROI,NeighborhoodSize,ExcitationCoordinates,Mpos,PlotSensorMap):
        ##### Here we build sensors locations and their respective feature map values:

        CoordinatesSet,LineWidth = FusionToolbox.CoordinatesFromROI(ROI)
        
        PathsSet = []       # Here we store all the paths with their corresponding values
        Image = cv2.GaussianBlur(Image,(NeighborhoodSize,NeighborhoodSize),cv2.BORDER_REFLECT_101)
        Image = Image - np.min(Image)
        Image = Image/np.max(Image)    
   
        for k in CoordinatesSet:
            PathsSet.append([k[0], k[1], Image[k[1], k[0]]])
        
        # Now lets arrange a set of paths with their respective values:
        PathsSet = np.array(PathsSet)
        sorted_indices = np.argsort(PathsSet[:, -1])    
        PathsSet_sorted = PathsSet[sorted_indices] 
        PathsSet_sorted = np.flip(PathsSet_sorted,axis = 0) 
        # Now we have a sorted set of paths starting from one showing the highest damage.
         
        ## BUILD OF A NEW FEATURE MAP
        
        PathImage = np.zeros(Image.shape)
        for ps in PathsSet_sorted:
            PathImage = cv2.line(PathImage,ExcitationCoordinates,[int(ps[0]),int(ps[1])],ps[2],LineWidth)
        
        kernel = np.ones((LineWidth, LineWidth), np.uint8) 
        # opening the image 
        PathImage = cv2.morphologyEx(PathImage, cv2.MORPH_CLOSE,kernel, iterations=1) 
        
        ## SHOW RESULTS
        
        if(PlotSensorMap):
            # Display a full (source) feature map
            FeatureMap = Image
            # Normalization is necessary for visualization purposes, 
            # to be decided if it should be used for all the calculations as well...
            # FeatureMap = FeatureMap - np.min(FeatureMap)
            # FeatureMap = FeatureMap/np.max(FeatureMap)   # This will serve for sensors' references 
            plt.figure()
            plt.title('Source feature map')
            plt.imshow(FeatureMap,vmin=0,vmax=1)
            plt.scatter(ExcitationCoordinates[0],ExcitationCoordinates[1],
                        s=200, facecolors = 'silver', edgecolors = 'black', marker = "h")
            plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
            for s in CoordinatesSet:
                plt.scatter(s[0],s[1],s=100, facecolors = 'none', edgecolors = 'black', marker = "o")
            plt.savefig('Figuredrop/aaa.png', dpi=400,bbox_inches='tight')    
            
            # Display simulated sensors used for calculations
            SensorsMap = np.zeros(Image.shape)           # This will be filled with sensor information
            for k in CoordinatesSet:
                SensorsMap = cv2.circle(SensorsMap,(k[0],k[1]),NeighborhoodSize,(FeatureMap[k[1],k[0]]),-1)
            plt.figure()
            plt.title('Sensor locations taken into consideration')
            plt.imshow(SensorsMap,vmin=0,vmax=1)
            plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
            plt.scatter(ExcitationCoordinates[0],ExcitationCoordinates[1],
                        s=200, facecolors = 'silver', edgecolors = 'black', marker = "h")      
            plt.savefig('Figuredrop/bbb.png', dpi=400,bbox_inches='tight')
            
            # Display feature map built from paths (reduced resolution)
            plt.figure()
            # PathImage = PathImage - np.min(PathImage)
            # PathImage = PathImage/np.max(PathImage) 
            plt.title('Feature map from composition of path values')
            plt.imshow(PathImage,vmin=0,vmax=1)
            plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
            plt.scatter(ExcitationCoordinates[0],ExcitationCoordinates[1],
                        s=200, facecolors = 'silver', edgecolors = 'black', marker = "h")  
            plt.savefig('Figuredrop/ccc.png', dpi=400,bbox_inches='tight')
        
   
        return PathImage, PathsSet_sorted, SensorsMap
    
    def plot_3D_surface(map2D, title='3D Damage Map'):
        # Force plot in external window without affecting other plots
        fig = plt.figure(figsize=(10, 8))
        # The key line below — show plot in standalone window
        plt.show(block=True)  # shows plot in a separate window *now*
        
        ax = fig.add_subplot(111, projection='3d')
    
        X = np.arange(map2D.shape[1])
        Y = np.arange(map2D.shape[0])
        X, Y = np.meshgrid(X, Y)
        Z = map2D
    
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=True)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Damage Probability')
    
        fig.colorbar(surf, shrink=0.5, aspect=10)
    
        # manager = plt.get_current_fig_manager()
        # try:
        #     manager.set_window_title(title)
        # except AttributeError:
        #     pass  # Backend doesn't support setting window title
        
    
    def PseudoRAPIDManySensors(Experiment, ArrayList, SourceList, Feature, ROI, Sigma, Mpos, PlotSensorMap, ToSave):
        # Sigma = 3  # Possibly later for addition in function heading
        Cmap = 'plasma'
        DamageOnlyInside = 0
        SaveFig = 1
    
        # Step 1: Get receiver coordinates from ROI
        CoordinatesSet, LineWidth = FusionToolbox.CoordinatesFromROI(ROI)
    
        # Step 2: Initialize probability map
        ProbabilityMap = np.zeros_like(ArrayList[0][0][1][Feature], dtype=np.float32)
    
        for idx, Measurement in enumerate(ArrayList):
            PartialImage = Measurement[0][1][Feature]
            PartialImage = PartialImage - np.min(PartialImage)
            PartialImage = PartialImage / np.max(PartialImage)
    
            # EC = FusionToolbox.PosS(SourceList[idx][0])  # Excitation coordinates
            EC = FusionToolbox.PosSensor(Experiment,SourceList[idx][0])
    
            for coord in CoordinatesSet:
                R = coord
                S = EC
    
                # Midpoint and direction vector
                M = [(R[0] + S[0]) / 2, (R[1] + S[1]) / 2]
                dx = R[0] - S[0]
                dy = R[1] - S[1]
                length = np.sqrt(dx ** 2 + dy ** 2)
    
                if length == 0:
                    continue  # Avoid division by zero
    
                direction = np.array([dx / length, dy / length])
    
                # Construct a 2D Gaussian around the midpoint, elongated along path
                for x in range(ProbabilityMap.shape[1]):
                    for y in range(ProbabilityMap.shape[0]):
                        r = np.array([x - M[0], y - M[1]])
                        dist_along = np.dot(r, direction)
            
                        # if np.abs(dist_along) > length / 2:
                        #     continue  # ⬅️ this line clips the Gaussian to the segment
            
                        dist_perp = np.linalg.norm(r - dist_along * direction)
            
                        sigma_parallel = length / 2
                        sigma_perpendicular = Sigma
                        
                        value = np.exp(
                            -0.5 * ((dist_along / sigma_parallel) ** 2 + (dist_perp / sigma_perpendicular) ** 2)
                            )
                        value *= PartialImage[int(R[1]), int(R[0])]
                        ProbabilityMap[y, x] += value
    
        # Normalize result
        ProbabilityMap = ProbabilityMap - np.min(ProbabilityMap)
        ProbabilityMap = ProbabilityMap / np.max(ProbabilityMap)
    
        # Optional masking
        if DamageOnlyInside:
            PossibleDamageMask = np.zeros(ProbabilityMap.shape)
            PossibleDamageMask = cv2.rectangle(PossibleDamageMask, [ROI[0][0], ROI[1][0]],
                                               [ROI[0][1], ROI[1][1]], 1, -1)
            ProbabilityMap *= PossibleDamageMask
    
        # Plotting
        plt.figure()
        plt.imshow(ProbabilityMap, cmap=Cmap, vmin=0, vmax=1)
        plt.scatter(Mpos[0], Mpos[1], s=100, facecolors='red', marker="x")
        
        ind = 0
        for k in ArrayList:
            if(Experiment == 1):
                EC = FusionToolbox.PosS(SourceList[ind][0])
            else:
                EC = FusionToolbox.PosSexp2(SourceList[ind][0])
            ind = ind + 1
            plt.scatter(EC[0],EC[1],s=200, facecolors = 'silver', edgecolors = 'black', marker = "h")
        
        
        # for src in SourceList:
        #     EC = FusionToolbox.PosS(src[0])
        #     plt.scatter(EC[0], EC[1], s=200, facecolors='silver', edgecolors='black', marker="h")
    
        if ToSave["DoWeSave"]:
            Name = ('Dmg_' + ToSave["damage"] + '_' + Feature +
                    '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) +
                    '_Sigma_' + str(Sigma) +
                    '_ROI_' + str(ToSave["ROI"][0]) +
                    '_RAPID_Fusion')
            plt.savefig(ToSave["FigureDrop"] + Name, dpi=400, bbox_inches='tight')
            
        # plt.figure()    
        # FusionToolbox.plot_3D_surface(ProbabilityMap, title='RAPID Output - 3D View')                                                                        
        return ProbabilityMap



    def AllMapManySensors(Experiment, ArrayList, SourceList, Feature, ROI, NeighborhoodSize, Mpos, PlotSensorMap, ToSave):
        Cmap = 'cividis'
        DamageOnlyInside = 1
        SaveFig = 1
        Sigma = 4
        ImageNormalize = 0


        # 1: Take information about ROI and build a set of corrdinates:
        CoordinatesSet, LineWidth = FusionToolbox.CoordinatesFromROI(ROI)
        # CoordinatesSet, LineWidth = FusionToolbox.CoordinatesWithinROI(ROI)
        # CoordinatesSet, LineWidth = FusionToolbox.CoordinatesEdgeBiasedROI(ROI)
        # CoordinatesSet, LineWidth = FusionToolbox.CoordinatesInGridROI(ROI)

        LineWidth = 2
        KernelSize = 5
        KernelSizeOpen = 3

       
       
        # 2. For all transmitters get values corresponding with receivers:
        PathsSet = []       
        ind = 0
        for k in ArrayList:

           PartialImage = k[0][1][Feature]
           if(ImageNormalize):
               # Here we are normalizing the feature image
               PartialImage = PartialImage - np.min(PartialImage)
               PartialImage = PartialImage/np.max(PartialImage)
           PartialImage = cv2.GaussianBlur(PartialImage,(NeighborhoodSize,NeighborhoodSize),
                                           cv2.BORDER_REFLECT_101)
           plt.figure()
           plt.imshow(PartialImage,vmin=0,vmax=1,cmap = Cmap)

           EC = FusionToolbox.PosSensor(Experiment,SourceList[ind][0])
           SensorTaken = SourceList[ind][0]
           if(ToSave["DoWeSave"]):
                    Name = ('Dmg_' + ToSave["damage"] + '_' + 
                            ToSave["Feature"] + '_F' + ToSave["Freq"] +
                            '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                            "_Neigh_" + str(ToSave["Neighborhood"]) +
                            '_ROI_' + str(ToSave["ROI"][0]) + 'PartialSensor' + str(SensorTaken) +
                            "InitialData")
                    plt.savefig(ToSave["FigureDrop"]+Name, dpi=200,bbox_inches='tight')
           
           ind = ind + 1
           
           SensorsMap = np.zeros(PartialImage.shape)
           for k in CoordinatesSet:
               SensorsMap = cv2.circle(SensorsMap,(k[0],k[1]),NeighborhoodSize,
                                       (PartialImage[k[1],k[0]]),-1)
           plt.figure()
           plt.title('Sensor locations taken into consideration')
           if(ImageNormalize):
               SensorsMap = SensorsMap - np.min(SensorsMap)
               SensorsMap = SensorsMap/np.max(SensorsMap) 
           for k in CoordinatesSet:
               PathsSet.append([k[0], k[1], EC[0], EC[1], SensorsMap[k[1], k[0]]])    
           
           SensorsMap = SensorsMap - np.min(SensorsMap)
           SensorsMap = SensorsMap/np.max(SensorsMap)     
           plt.imshow(SensorsMap,vmin=0,vmax=1)
           plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
           plt.scatter(EC[0],EC[1], s=200, facecolors = 'silver', edgecolors = 'black', marker = "h") 
           if(ToSave["DoWeSave"]):
               Name = ('Dmg_' + ToSave["damage"] + '_' + 
                       ToSave["Feature"] + '_F' + ToSave["Freq"] +
                       '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                       "_Neigh_" + str(ToSave["Neighborhood"]) +
                       '_ROI_' + str(ToSave["ROI"][0]) + 'PartialSensor' + str(SensorTaken) +
                       "ResultsOfBorderFusion")
               plt.savefig(ToSave["FigureDrop"]+Name, dpi=400,bbox_inches='tight')

           


        # Now lets arrange a set of paths with their respective values:
        PathsSet = np.array(PathsSet)
        sorted_indices = np.argsort(PathsSet[:, -1])    
        PathsSet_sorted = PathsSet[sorted_indices] 
        PathsSet_sorted = np.flip(PathsSet_sorted,axis = 0) 
        
       # 3. BUILD A NEW FEATURE MAP - using Image fusion approach

        PathImage = np.zeros(PartialImage.shape)
        for ps in PathsSet_sorted:
            PathImage = cv2.line(PathImage,[int(ps[2]),int(ps[3])],[int(ps[0]),int(ps[1])],ps[4],LineWidth)
        # plt.figure()
        # plt.imshow(PathImage,vmin=0,vmax=1,cmap = Cmap)
        kernel = np.ones((KernelSize, KernelSize), np.uint8) 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KernelSize,KernelSize))
        kernelOpen = np.ones((KernelSizeOpen, KernelSizeOpen), np.uint8) 
        # kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KernelSizeOpen,KernelSizeOpen))
        # opening the image - we don't do that because we have enough paths now
        PathImage = cv2.morphologyEx(PathImage, cv2.MORPH_CLOSE,kernel, iterations=1)
        PathImage = cv2.morphologyEx(PathImage, cv2.MORPH_OPEN,kernelOpen, iterations=1)
        PathImageNorm = PathImage/np.max(PathImage)
        
        if(DamageOnlyInside):
            PossibleDamageMask = np.zeros(PartialImage.shape)
            PossibleDamageMask = cv2.rectangle(PossibleDamageMask, [ROI[0][0], ROI[1][0]],
                                               [ROI[0][1],ROI[1][1]], 1, -1)
            PathImage = PathImage*PossibleDamageMask
            PathImageNorm = PathImageNorm*PossibleDamageMask
    
        plt.figure()
        plt.imshow(PathImage,vmin=0,vmax=1,cmap = Cmap)
        plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
        ind = 0
        for k in ArrayList:
            if(Experiment == 1):
                EC = FusionToolbox.PosS(SourceList[ind][0])
            else:
                EC = FusionToolbox.PosSexp2(SourceList[ind][0])
            ind = ind + 1
            plt.scatter(EC[0],EC[1],s=200, facecolors = 'silver', edgecolors = 'black', marker = "h")
        if(ToSave["DoWeSave"]):
            Name = ('Dmg_' + ToSave["damage"] + '_' + ToSave["Feature"] + '_F' + ToSave["Freq"] +
                    '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                    'LineWidth' + str(LineWidth) + 
                    "_Neigh_" + str(ToSave["Neighborhood"]) +
                    '_ROI_' + str(ToSave["ROI"][0]) + 
                    "ResultsOfBorderFusion")
            plt.savefig(ToSave["FigureDrop"]+Name, dpi=400,bbox_inches='tight')
            
        plt.figure()
        plt.imshow(PathImageNorm,vmin=0,vmax=1,cmap = Cmap)
        plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
        ind = 0
        for k in ArrayList:
             if(Experiment == 1):
                 EC = FusionToolbox.PosS(SourceList[ind][0])
             else:
                 EC = FusionToolbox.PosSexp2(SourceList[ind][0])
             ind = ind + 1
             plt.scatter(EC[0],EC[1],s=200, facecolors = 'silver', edgecolors = 'black', marker = "h")
        if(ToSave["DoWeSave"]):
             Name = ('Dmg_' + ToSave["damage"] + '_' + ToSave["Feature"] + '_F' + ToSave["Freq"] +
                     '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                     'LineWidth' + str(LineWidth) + 
                     "_Neigh_" + str(ToSave["Neighborhood"]) +
                     '_ROI_' + str(ToSave["ROI"][0]) + 
                     "ResultsOfBorderFusionNormalized")
             plt.savefig(ToSave["FigureDrop"]+Name, dpi=400,bbox_inches='tight')    
         
         
           
    # 4. BUILD A NEW FEATURE MAP: Using RAPID approach (pseudo-RAPID)
    
        # RAPIDImage = np.zeros(PartialImage.shape)

        # for ps in PathsSet_sorted:
        #     R = np.array([ps[0], ps[1]])
        #     S = np.array([ps[2], ps[3]])
        #     DI_value = ps[4]

        #     # Midpoint and direction vector
        #     M = (R + S) / 2
        #     path_vector = R - S
        #     length = np.linalg.norm(path_vector)
        #     if length == 0:
        #         continue
        #     u = path_vector / length  # unit direction vector

        #     sigma_parallel = length / 2  # along-path Gaussian spread
        #     sigma_perpendicular = Sigma  # across-path spread

        #     for y in range(RAPIDImage.shape[0]):
        #         for x in range(RAPIDImage.shape[1]):
        #             P = np.array([x, y])
        #             r = P - M
        #             dist_along = np.dot(r, u)
        #             dist_perp = np.linalg.norm(r - dist_along * u)

        #             # Elliptical Gaussian weight centered on midpoint
        #             value = np.exp(
        #                 -0.5 * ((dist_along / sigma_parallel) ** 2 + (dist_perp / sigma_perpendicular) ** 2)
        #             )
        #             value *= DI_value
        #             RAPIDImage[y, x] += value
        # RAPIDImageNorm = RAPIDImage / np.max(RAPIDImage)
        # if(DamageOnlyInside):
        #     PossibleDamageMask = np.zeros(PartialImage.shape)
        #     PossibleDamageMask = cv2.rectangle(PossibleDamageMask, [ROI[0][0], ROI[1][0]],
        #                                        [ROI[0][1], ROI[1][1]], 1, -1)
        #     RAPIDImage = RAPIDImage * PossibleDamageMask
        #     RAPIDImageNorm = RAPIDImageNorm * PossibleDamageMask
           
           
        # plt.figure()
        # plt.imshow(RAPIDImage, vmin=0, vmax=1, cmap=Cmap)
        # plt.title('Pseudo RAPID Map')
        # plt.scatter(Mpos[0], Mpos[1], s=100, facecolors='red', marker="x")
        # ind = 0
        # for k in ArrayList:
        #     if Experiment == 1:
        #         EC = FusionToolbox.PosS(SourceList[ind][0])
        #     else:
        #         EC = FusionToolbox.PosSexp2(SourceList[ind][0])
        #     ind += 1
        #     plt.scatter(EC[0], EC[1], s=200, facecolors='silver', edgecolors='black', marker="h")
    
        # if(ToSave["DoWeSave"]):
        #     Name = ('Dmg_' + ToSave["damage"] + '_' + ToSave["Feature"] + '_F' + ToSave["Freq"] +
        #             '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
        #             'LineWidth' + str(LineWidth) + 
        #             "_Neigh_" + str(ToSave["Neighborhood"]) +
        #             '_ROI_' + str(ToSave["ROI"][0]) + 
        #             "ResultsOfPseudoRAPID")
        #     plt.savefig(ToSave["FigureDrop"]+Name, dpi=400, bbox_inches='tight')
            
            
        # plt.figure()
        # plt.imshow(RAPIDImageNorm, vmin=0, vmax=1, cmap=Cmap)
        # plt.title('Pseudo RAPID Map')
        # plt.scatter(Mpos[0], Mpos[1], s=100, facecolors='red', marker="x")
        # ind = 0
        # for k in ArrayList:
        #      if Experiment == 1:
        #          EC = FusionToolbox.PosS(SourceList[ind][0])
        #      else:
        #          EC = FusionToolbox.PosSexp2(SourceList[ind][0])
        #      ind += 1
        #      plt.scatter(EC[0], EC[1], s=200, facecolors='silver', edgecolors='black', marker="h")
    
        # if(ToSave["DoWeSave"]):
        #      Name = ('Dmg_' + ToSave["damage"] + '_' + ToSave["Feature"] + '_F' + ToSave["Freq"] +
        #              '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
        #              'LineWidth' + str(LineWidth) + 
        #              "_Neigh_" + str(ToSave["Neighborhood"]) +
        #              '_ROI_' + str(ToSave["ROI"][0]) + 
        #              "ResultsOfPseudoRAPIDNormalized")
        #      plt.savefig(ToSave["FigureDrop"]+Name, dpi=400, bbox_inches='tight')
         
        
        
        
    
        
    # 5. BUILD A NEW FEATURE MAP: Using True RAPID approach 
        # print("DEBUG: I'm starting RAPID now")
        beta  = 1.1          # 0 < beta < 1  → smaller -> sharper ellipse
        alpha = 1.0           # raise DI to emphasise strong paths (α≈1–2)
        eps   = 1e-12         # protects against division-by-zero
        
        # Work images -------------------------------------------------------------
        RAPIDImage   = np.zeros_like(PartialImage, dtype=float)
        KernelSum    = np.zeros_like(PartialImage, dtype=float)
        
        # Main loop over the path set --------------------------------------------
        for ps in PathsSet_sorted:
            # Rx,Ry,Sx,Sy already in pixel coordinates of PartialImage
            R = np.array([ps[0], ps[1]], dtype=float)
            S = np.array([ps[2], ps[3]], dtype=float)
            DI_value = ps[4]
        
            # Skip degenerate paths
            length = np.linalg.norm(R - S)
            

            # print("DEBUG: R,S,length,beta:",R, S, length, beta)
            
            if length == 0:
                continue
        
            # ---------------------------------------------------------------------
            # loop over every pixel (x,y) and update the image
            # ---------------------------------------------------------------------
            for y in range(RAPIDImage.shape[0]):
                for x in range(RAPIDImage.shape[1]):
                    P = np.array([x, y], dtype=float)
        
                    # Ellipse indicator R_xy  (Van Velsor 2007)
                    path_len_sum = np.linalg.norm(P - R) + np.linalg.norm(P - S)
                    R_xy = path_len_sum / length            # ≥1 everywhere, =1 on the path
        
                    if R_xy < beta:                         # only inside the ellipse
                        weight = (beta - R_xy) / (beta - 1.0) # linear taper 1→0
                    else:
                        continue                            # outside ellipse → contribute nothing
        
                    # update numerator and denominator
                    RAPIDImage[y, x] += (DI_value ** alpha) * weight
                    KernelSum[y, x]  += weight

            # print("First path – sum of weights:",KernelSum.sum())
        # -------------------------------------------------------------------------
        # final normalisation (probability map 0–1)
        # -------------------------------------------------------------------------
        # RAPIDImageNorm = RAPIDImage / (KernelSum + eps)      # unbiased fusion
        # RAPIDImageNorm -= RAPIDImageNorm.min()               # scale 0 – 1
        # RAPIDImageNorm /= (RAPIDImageNorm.max() + eps)
        
        RAPIDImageNorm = RAPIDImage - np.min(RAPIDImage)
        RAPIDImageNorm = RAPIDImageNorm/np.max(RAPIDImageNorm)
        
        # RAPIDImageNorm = RAPIDImage / np.max(RAPIDImage)
        
        # RAPIDImageNorm = RAPIDImage / np.max(RAPIDImage)
        
        # optional ROI masking exactly as you had it ------------------------------
        if DamageOnlyInside:
            PossibleDamageMask = np.zeros_like(PartialImage, dtype=float)
            PossibleDamageMask = cv2.rectangle(
                PossibleDamageMask,
                [ROI[0][0], ROI[1][0]],
                [ROI[0][1], ROI[1][1]],
                1, -1
            )
            RAPIDImage       *= PossibleDamageMask
            RAPIDImageNorm   *= PossibleDamageMask
            
   
        plt.figure()
        plt.imshow(RAPIDImage, vmin=0, vmax=1, cmap=Cmap)
        plt.title('RAPID Map')
        plt.scatter(Mpos[0], Mpos[1], s=100, facecolors='red', marker="x")
        ind = 0
        for k in ArrayList:
            if Experiment == 1:
                EC = FusionToolbox.PosS(SourceList[ind][0])
            else:
                EC = FusionToolbox.PosSexp2(SourceList[ind][0])
            ind += 1
            plt.scatter(EC[0], EC[1], s=200, facecolors='silver', edgecolors='black', marker="h")
    
        if(ToSave["DoWeSave"]):
            Name = ('Dmg_' + ToSave["damage"] + '_' + ToSave["Feature"] + '_F' + ToSave["Freq"] +
                    '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                    'LineWidth' + str(LineWidth) + 
                    "_Neigh_" + str(ToSave["Neighborhood"]) +
                    '_ROI_' + str(ToSave["ROI"][0]) + 
                    "ResultsOfRAPID")
            plt.savefig(ToSave["FigureDrop"]+Name, dpi=400, bbox_inches='tight')
            
            
        plt.figure()
        plt.imshow(RAPIDImageNorm, vmin=0, vmax=1, cmap=Cmap)
        plt.title('RAPID Map')
        plt.scatter(Mpos[0], Mpos[1], s=100, facecolors='red', marker="x")
        ind = 0
        for k in ArrayList:
             if Experiment == 1:
                 EC = FusionToolbox.PosS(SourceList[ind][0])
             else:
                 EC = FusionToolbox.PosSexp2(SourceList[ind][0])
             ind += 1
             plt.scatter(EC[0], EC[1], s=200, facecolors='silver', edgecolors='black', marker="h")
    
        if(ToSave["DoWeSave"]):
             Name = ('Dmg_' + ToSave["damage"] + '_' + ToSave["Feature"] + '_F' + ToSave["Freq"] +
                     '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                     'LineWidth' + str(LineWidth) + 
                     "_Neigh_" + str(ToSave["Neighborhood"]) +
                     '_ROI_' + str(ToSave["ROI"][0]) + 
                     "ResultsOfRAPIDNormalized")
             plt.savefig(ToSave["FigureDrop"]+Name, dpi=400, bbox_inches='tight')
         
                    
            


        return

    def BorderReductionManySensors(ArrayList,SourceList,Feature,
                                   ROI,NeighborhoodSize,Mpos,PlotSensorMap,ToSave):
        # LineWidth= 1
        Cmap = 'viridis'
        DamageOnlyInside = 1
        SaveFig = 1

 
        # 1: Take information about ROI and build a set of corrdinates:
        CoordinatesSet, LineWidth = FusionToolbox.CoordinatesFromROI(ROI)
        # CoordinatesSet, LineWidth = FusionToolbox.CoordinatesWithinROI(ROI)
        # CoordinatesSet, LineWidth = FusionToolbox.CoordinatesEdgeBiasedROI(ROI)
        # CoordinatesSet, LineWidth = FusionToolbox.CoordinatesInGridROI(ROI)
        LineWidth = 2
        # LineWidth = ToSave["LineWidth"]
        KernelSize = 5
        KernelSizeOpen = 3
        # 2. For all transmitters get values corresponding with receivers:
        PathsSet = []       
        ind = 0
        for k in ArrayList:
            PartialImage = k[0][1][Feature]
            PartialImage = PartialImage - np.min(PartialImage)
            PartialImage = PartialImage/np.max(PartialImage) 
            PartialImage = cv2.GaussianBlur(PartialImage,(NeighborhoodSize,NeighborhoodSize),
                                            cv2.BORDER_REFLECT_101)
            plt.figure()
            plt.imshow(PartialImage,vmin=0,vmax=1,cmap = Cmap)

            EC = FusionToolbox.PosS(SourceList[ind][0])
            SensorTaken = SourceList[ind][0]
            if(ToSave["DoWeSave"]):
                     Name = ('Dmg_' + ToSave["damage"] + '_' + 'PartialSensor' + str(SensorTaken) +
                             ToSave["Feature"] + 
                             '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                             "_Neigh_" + str(ToSave["Neighborhood"]) +
                             '_ROI_' + str(ToSave["ROI"][0]) + 
                             "InitialData")
                     plt.savefig(ToSave["FigureDrop"]+Name, dpi=200,bbox_inches='tight')
            
            ind = ind + 1
            
            SensorsMap = np.zeros(PartialImage.shape)
            for k in CoordinatesSet:
                SensorsMap = cv2.circle(SensorsMap,(k[0],k[1]),NeighborhoodSize,
                                        (PartialImage[k[1],k[0]]),-1)
            plt.figure()
            plt.title('Sensor locations taken into consideration')
            SensorsMap = SensorsMap - np.min(SensorsMap)
            SensorsMap = SensorsMap/np.max(SensorsMap) 
            plt.imshow(SensorsMap,vmin=0,vmax=1)
            plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
            plt.scatter(EC[0],EC[1], s=200, facecolors = 'silver', edgecolors = 'black', marker = "h") 
            if(ToSave["DoWeSave"]):
                Name = ('Dmg_' + ToSave["damage"] + '_' + 'PartialSensor' + str(SensorTaken) +
                        ToSave["Feature"] + 
                        '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                        "_Neigh_" + str(ToSave["Neighborhood"]) +
                        '_ROI_' + str(ToSave["ROI"][0]) + 
                        "ResultsOfBorderFusion")
                plt.savefig(ToSave["FigureDrop"]+Name, dpi=400,bbox_inches='tight')

            
            for k in CoordinatesSet:
                PathsSet.append([k[0], k[1], EC[0], EC[1], SensorsMap[k[1], k[0]]])

        # Now lets arrange a set of paths with their respective values:
        PathsSet = np.array(PathsSet)
        sorted_indices = np.argsort(PathsSet[:, -1])    
        PathsSet_sorted = PathsSet[sorted_indices] 
        PathsSet_sorted = np.flip(PathsSet_sorted,axis = 0) 
        
        # BUILD A NEW FEATURE MAP:

        PathImage = np.zeros(PartialImage.shape)
        for ps in PathsSet_sorted:
            PathImage = cv2.line(PathImage,[int(ps[2]),int(ps[3])],[int(ps[0]),int(ps[1])],ps[4],LineWidth)
        # plt.figure()
        # plt.imshow(PathImage,vmin=0,vmax=1,cmap = Cmap)
        kernel = np.ones((KernelSize, KernelSize), np.uint8) 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KernelSize,KernelSize))
        kernelOpen = np.ones((KernelSizeOpen, KernelSizeOpen), np.uint8) 
        # kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KernelSizeOpen,KernelSizeOpen))
        # opening the image - we don't do that because we have enough paths now
        PathImage = cv2.morphologyEx(PathImage, cv2.MORPH_CLOSE,kernel, iterations=1)
        PathImage = cv2.morphologyEx(PathImage, cv2.MORPH_OPEN,kernelOpen, iterations=1)
        if(DamageOnlyInside):
            PossibleDamageMask = np.zeros(PartialImage.shape)
            PossibleDamageMask = cv2.rectangle(PossibleDamageMask, [ROI[0][0], ROI[1][0]],
                                               [ROI[0][1],ROI[1][1]], 1, -1)
            PathImage = PathImage*PossibleDamageMask
 
        plt.figure()
        plt.imshow(PathImage,vmin=0,vmax=1,cmap = Cmap)
        plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
        ind = 0
        for k in ArrayList:
            EC = FusionToolbox.PosS(SourceList[ind][0])
            ind = ind + 1
            plt.scatter(EC[0],EC[1],s=200, facecolors = 'silver', edgecolors = 'black', marker = "h")
        
        
        # ToSave = {"damage": "11",
        #           "Feature": Feature,
        #           "Neighborhood": NeighborhoodSize
        #           "Window": Window,
        #           "Freq": ExcitationFreq,
        #           "ROI": ROI,
        #           "Sources": [sublist[0] for sublist in SourceList],
        #           "DmgPos": Mpos,
        #           "FigureDrop":SaveDrop}
        
        
        if(ToSave["DoWeSave"]):
            Name = ('Dmg_' + ToSave["damage"] + '_' + ToSave["Feature"] + 
                    '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                    'LineWidth' + str(LineWidth) + 
                    "_Neigh_" + str(ToSave["Neighborhood"]) +
                    '_ROI_' + str(ToSave["ROI"][0]) + 
                    "ResultsOfBorderFusion")
            plt.savefig(ToSave["FigureDrop"]+Name, dpi=400,bbox_inches='tight')



        return
    
    
    def BorderReductionManySensorsGeneralized(Experiment, ArrayList,SourceList,Feature,
                                   ROI,NeighborhoodSize,Mpos,PlotSensorMap,ToSave):
        # LineWidth= 1
        Cmap = 'viridis'
        DamageOnlyInside = 1
        SaveFig = 1

 
        # 1: Take information about ROI and build a set of corrdinates:
        CoordinatesSet, LineWidth = FusionToolbox.CoordinatesFromROI(ROI)
        LineWidth = 2
        # LineWidth = ToSave["LineWidth"]
        KernelSize = 5
        KernelSizeOpen = 3
        # 2. For all transmitters get values corresponding with receivers:
        PathsSet = []       
        ind = 0
        for k in ArrayList:
            # PartialImage = k[0][1][Feature]
            PartialImage = k[1][Feature]
            PartialImage = PartialImage - np.min(PartialImage)
            PartialImage = PartialImage/np.max(PartialImage) 
            PartialImage = cv2.GaussianBlur(PartialImage,(NeighborhoodSize,NeighborhoodSize),
                                            cv2.BORDER_REFLECT_101)
            plt.figure()
            plt.imshow(PartialImage,vmin=0,vmax=1,cmap = Cmap)

            EC = FusionToolbox.PosSensor(Experiment,SourceList[ind][0])
            SensorTaken = SourceList[ind][0]
            if(ToSave["DoWeSave"]):
                     Name = ('Dmg_' + ToSave["damage"] + '_' + 'PartialSensor' + str(SensorTaken) +
                             ToSave["Feature"] + 
                             '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                             "_Neigh_" + str(ToSave["Neighborhood"]) +
                             '_ROI_' + str(ToSave["ROI"][0]) + 
                             "InitialData")
                     plt.savefig(ToSave["FigureDrop"]+Name, dpi=200,bbox_inches='tight')
            
            ind = ind + 1
            
            SensorsMap = np.zeros(PartialImage.shape)
            for k in CoordinatesSet:
                SensorsMap = cv2.circle(SensorsMap,(k[0],k[1]),NeighborhoodSize,
                                        (PartialImage[k[1],k[0]]),-1)
            plt.figure()
            plt.title('Sensor locations taken into consideration')
            SensorsMap = SensorsMap - np.min(SensorsMap)
            SensorsMap = SensorsMap/np.max(SensorsMap) 
            plt.imshow(SensorsMap,vmin=0,vmax=1)
            plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
            plt.scatter(EC[0],EC[1], s=200, facecolors = 'silver', edgecolors = 'black', marker = "h") 
            if(ToSave["DoWeSave"]):
                Name = ('Dmg_' + ToSave["damage"] + '_' + 'PartialSensor' + str(SensorTaken) +
                        ToSave["Feature"] + 
                        '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                        "_Neigh_" + str(ToSave["Neighborhood"]) +
                        '_ROI_' + str(ToSave["ROI"][0]) + 
                        "ResultsOfBorderFusion")
                plt.savefig(ToSave["FigureDrop"]+Name, dpi=400,bbox_inches='tight')

            
            for k in CoordinatesSet:
                PathsSet.append([k[0], k[1], EC[0], EC[1], SensorsMap[k[1], k[0]]])

        # Now lets arrange a set of paths with their respective values:
        PathsSet = np.array(PathsSet)
        sorted_indices = np.argsort(PathsSet[:, -1])    
        PathsSet_sorted = PathsSet[sorted_indices] 
        PathsSet_sorted = np.flip(PathsSet_sorted,axis = 0) 
        
        # BUILD A NEW FEATURE MAP:

        PathImage = np.zeros(PartialImage.shape)
        for ps in PathsSet_sorted:
            PathImage = cv2.line(PathImage,[int(ps[2]),int(ps[3])],[int(ps[0]),int(ps[1])],ps[4],LineWidth)
        # plt.figure()
        # plt.imshow(PathImage,vmin=0,vmax=1,cmap = Cmap)
        kernel = np.ones((KernelSize, KernelSize), np.uint8) 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KernelSize,KernelSize))
        kernelOpen = np.ones((KernelSizeOpen, KernelSizeOpen), np.uint8) 
        # kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KernelSizeOpen,KernelSizeOpen))
        # opening the image - we don't do that because we have enough paths now
        PathImage = cv2.morphologyEx(PathImage, cv2.MORPH_CLOSE,kernel, iterations=1)
        PathImage = cv2.morphologyEx(PathImage, cv2.MORPH_OPEN,kernelOpen, iterations=1)
        if(DamageOnlyInside):
            PossibleDamageMask = np.zeros(PartialImage.shape)
            PossibleDamageMask = cv2.rectangle(PossibleDamageMask, [ROI[0][0], ROI[1][0]],
                                               [ROI[0][1],ROI[1][1]], 1, -1)
            PathImage = PathImage*PossibleDamageMask
 
        plt.figure()
        plt.imshow(PathImage,vmin=0,vmax=1,cmap = Cmap)
        plt.scatter(Mpos[0],Mpos[1],s=100, facecolors = 'red', marker = "x")
        ind = 0
        for k in ArrayList:
            if(Experiment == 1):
                EC = FusionToolbox.PosS(SourceList[ind][0])
            else:
                EC = FusionToolbox.PosSexp2(SourceList[ind][0])
            ind = ind + 1
            plt.scatter(EC[0],EC[1],s=200, facecolors = 'silver', edgecolors = 'black', marker = "h")
        
        
        # ToSave = {"damage": "11",
        #           "Feature": Feature,
        #           "Neighborhood": NeighborhoodSize
        #           "Window": Window,
        #           "Freq": ExcitationFreq,
        #           "ROI": ROI,
        #           "Sources": [sublist[0] for sublist in SourceList],
        #           "DmgPos": Mpos,
        #           "FigureDrop":SaveDrop}
        
        
        if(ToSave["DoWeSave"]):
            Name = ('Dmg_' + ToSave["damage"] + '_' + ToSave["Feature"] + 
                    '_W' + str(ToSave["Window"][0]) + 'to' + str(ToSave["Window"][1]) + 
                    'LineWidth' + str(LineWidth) + 
                    "_Neigh_" + str(ToSave["Neighborhood"]) +
                    '_ROI_' + str(ToSave["ROI"][0]) + 
                    "ResultsOfBorderFusion")
            plt.savefig(ToSave["FigureDrop"]+Name, dpi=400,bbox_inches='tight')



        return
    
    
    
    
    def CalculateMeasures(Image,Mpos):
         Cmap = 'cividis'
         IM = Image
         IM2= Image
         NeighborhoodRadius = 8;
         MaxIndex = np.unravel_index(IM.argmax(),IM.shape)
         mask = np.zeros(IM.shape,dtype = np.uint8)
         mask = cv2.circle(mask,(Mpos[0],Mpos[1]),NeighborhoodRadius,(255,255,255),-1)
         mask2 = np.zeros(IM2.shape,dtype = np.uint8)
         mask2 = cv2.circle(mask2,(MaxIndex[1],MaxIndex[0]),NeighborhoodRadius,(255,255,255),-1)
         negativemask = abs(255-mask)
         negativemask2 = abs(255-mask2)
         neighborhood = (IM*mask/255)
         neighborhood2 = (IM2*mask2/255)
         background = (IM*negativemask/255)
         background2 = (IM2*negativemask2/255)
         # plt.figure()
         # plt.imshow(background,cmap = Cmap)
         # plt.colorbar()
              # plt.imshow(neighborhood)
         
         D = np.sqrt((Mpos[0]-MaxIndex[1])*(Mpos[0]-MaxIndex[1])+(Mpos[1]-MaxIndex[0])*(Mpos[1]-MaxIndex[0]))
         
         N = (sum(sum(neighborhood))/np.count_nonzero(neighborhood))/(sum(sum(background))/np.count_nonzero(background))
    
         Nprime = (sum(sum(neighborhood2))/np.count_nonzero(neighborhood2))/(sum(sum(background2))/np.count_nonzero(background2))
    
         return D,N,Nprime
  
    def ShowAndFuseComponents(ArrayList,Feature,ShowPlot,Gauss,GM,Mpos,SaveFig,SourceList,DmInf,ResolutionReductionFactor,SaveDrop):
        # This function takes a list of measurement arrays, filters them and then fuses them together showing all the results
        # Cmap = 'gist_ncar'
        Cmap = 'cividis'
        index = 1
        FusedResult = []
        MaxAmp = []
        MeanAmp = []
        for k in ArrayList:
            FuseComponent = k[0][1][Feature];
            if(ResolutionReductionFactor > 1):
                FuseComponent = FusionToolbox.ReduceResolution(FuseComponent,ResolutionReductionFactor)
            FuseComponent = cv2.GaussianBlur(FuseComponent,(Gauss,Gauss),cv2.BORDER_REFLECT_101)
            MaxAmp.append(np.max(FuseComponent))
            MeanAmp.append(np.mean(FuseComponent))
            if(GM == 1):
                FuseComponent = FuseComponent - np.min(FuseComponent)
                # FuseComponent = FuseComponent/np.percentile(FuseComponent,75)
                FuseComponent = FuseComponent/np.max(FuseComponent)
            if(ShowPlot):
                plt.figure()
                plt.imshow(FuseComponent,cmap = Cmap)
                plt.colorbar()
                # plt.scatter(Mpos[0],Mpos[1],s=300, facecolors = 'green', edgecolors = 'blue', marker = "+")
                MaxIndex = np.unravel_index(FuseComponent.argmax(),FuseComponent.shape)
                # plt.scatter(MaxIndex[1],MaxIndex[0],s=200, facecolors = 'cyan', edgecolors = 'blue', marker = "x")
                D,N,NPrime = FusionToolbox.CalculateMeasures(FuseComponent,Mpos)
                D = round(D,2)
                N = round(N,3)
                plt.title(DmInf+'mm '+'S'+str(SourceList[index-1][0])+'_'+str(SourceList[index-1][1])+'to'+str(SourceList[index-1][2])+', ' + str(Feature)+', D='+str(D)+', N = '+str(N), y=1.08)
                plt.gcf().set_size_inches(3.1, 3.1)
                if(SaveFig):
                    Name = 'Dmg_' + DmInf + '_' + str(Feature) +'_Gauss_'+str(Gauss) + '_S_' + str(SourceList[index-1][0])+'_M_'+ str(SourceList[index-1][1]) +'_to_'+ str(SourceList[index-1][2]) 
                    plt.savefig(SaveDrop+Name, dpi=400,bbox_inches='tight')
       
            if(index == 1):
                FusedResult = FuseComponent
            else:
                # FusedResult + FusedResult + FuseComponent    
                FusedResult = FusedResult + FuseComponent 
            index = index + 1
    
            print("Component measures, dist: %5.2f, noise: %5.3f, max amp:%5.3f,mean amp:%5.3f" % (D,N,MaxAmp[-1],MeanAmp[-1]))
        FusedResult = FusedResult/(index-1)
        MaxAmp.append(np.mean(MaxAmp))
        MeanAmp.append(np.mean(MeanAmp))
        if(GM == 1):
            FusedResult = FusedResult - np.min(FusedResult)
             # FuseComponent = FuseComponent/np.percentile(FuseComponent,75)
            FusedResult = FusedResult/np.max(FusedResult)
        plt.figure()
        plt.imshow(FusedResult,cmap = Cmap)
        D,N,NPrime = FusionToolbox.CalculateMeasures(FusedResult,Mpos)
        D = round(D,2)
        N = round(N,3)
        plt.title('FusedResult, ' +Feature+', D='+str(D)+', N = '+str(N), y=1.08)
        plt.colorbar()
        # plt.scatter(Mpos[0],Mpos[1],s=1000, facecolors = 'red', edgecolors = 'blue', marker = "+")
        MaxIndex = np.unravel_index(FusedResult.argmax(),FusedResult.shape)
        # plt.scatter(MaxIndex[1],MaxIndex[0],s=200, facecolors = 'magenta', edgecolors = 'blue', marker = "x")
        plt.gcf().set_size_inches(3.1, 3.1)
    
        print("Fused measures, dist: %5.2f, noise: %5.3f, mean max amp:%5.3f, mean mean amp: :%5.3f" % (D,N,MaxAmp[-1],MeanAmp[-1]))
        print("Nprime =  %5.3f" % (NPrime))
        
        if(SaveFig):
             Name = 'Dmg_' + DmInf + '_' + str(Feature)+'_Gauss_'+str(Gauss)+'_Fusion'
             plt.savefig(SaveDrop+Name, dpi=400,bbox_inches='tight')
        # figure,axes = plt.subplots()
        # cc = plt.Circle((0.4,0.6),0.4)
        # axes.set_aspect(1)
        # axes.add_artist(cc)
        # plt.show()
        return FusedResult
    
    