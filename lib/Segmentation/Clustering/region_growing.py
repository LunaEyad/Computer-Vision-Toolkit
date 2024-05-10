import cv2
import numpy as np
import itertools
from skimage.feature import peak_local_max


class GrowRegion():
    """
    Class for handling region growing algorithm.
    """
    def __init__(self, img_path, threshold, flag, connect):
        """
        Initializes the GrowRegion class with the given image path, threshold, image color flag, and connectivity flag.

        Args:
            img_path (str): The path to the image.
            threshold (float): The threshold value for region growing.
            flag (int): 0 for grayscale, 1 for color.
            connect (int): 0 for 4-connected, 1 for 8-connected.
        """
        self.path = img_path
        self.flag = flag
        self.readImage() # read image 
        # get image height and width
        self.h = self.img.shape[0]
        self.w = self.img.shape[1]
        self.thresh = threshold
        self.connect = connect
        # create 2D array of zeros to keep track of whether pixels have been visited during region growing
        self.visited = np.zeros((self.h, self.w), np.double)
        # create 3D array (RGB image) of zeros to store segmented regions
        self.segments = np.zeros((self.h, self.w, 3), dtype='uint8')
        # initialize stack object to be used in region growing
        self.stack = Stack()
        self.seeds = [] # initialize seeds list
        self.currentRegion = 0 # initialize the current region number to zero
        self.iterations = 0 # initialize iteration count to zero
        
    def readImage(self):
        """
        Reads the image from the given path. If flag is set, reads the image as RGB, otherwise reads it as grayscale.

        Args:
            img_path (str): The path to the image.
        """
        if self.flag:
            # Read image as RGB
            self.img = cv2.imread(self.path, 1).astype('int')
        else: 
            # Read image as Grayscale
            self.img = cv2.imread(self.path)
            src = np.copy(self.img)
            color_img = cv2.cvtColor(src, cv2.COLOR_Luv2BGR)
            img_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY) 
            self.img = img_gray           

    def getNeighbor(self, x0, y0) -> list:
        """
        Gets the neighbors of the given pixel. If connect is set, gets the 8-connected neighbors, otherwise gets the 4-connected neighbors.

        Args:
            x0 (int): The x-coordinate of the pixel (seed point).
            y0 (int): The y-coordinate of the pixel (seed point).

        Returns:
            list: A list of tuples representing the coordinates of the neighboring pixels of the seed point.
        """
        if self.connect:
            # 8-connected neighborhood
            return [
                # initialize neighbor pixel tuple/coordinate
                (x, y)
                # Generate all possible pairs of values from the Cartesian product of the set (-1, 0, 1) repeated twice
                # list of tuples representing the relative offsets for neighboring pixels
                # (-1, -1) (-1, 0) (-1, 1) (0, -1) (0, 0) (0, 1) (1, -1) (1, 0) (1, 1)
                for i, j in itertools.product((-1, 0, 1), repeat=2)
                # forgo center pixel and check if neighboring pixels are within image boundaries
                if (i, j) != (0, 0) and self.imageBoundaries(x := x0 + i, y := y0 + j)
            ]
        else:
            # 4-connected neighborhood
            return [
                # initialize neighbor pixel tuple/coordinate
                (x, y)
                # get the offsets of the 4 neighbor pixels: (-1, 0) (0, -1) (0, 1) (1, 0)
                for i, j in [(0, -1), (-1, 0), (1, 0), (0, 1)]
                # check if neighboring pixels are within image boundaries
                if self.imageBoundaries(x := x0 + i, y := y0 + j)
            ]
    
    def imageBoundaries(self, x, y) -> bool:
        """
        Checks if the given coordinates are within the image boundaries.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.

        Returns:
            bool: True if the coordinates are within the image boundaries, False otherwise.
        """
        return  0<=x<self.h and 0<=y<self.w
    
    def generateSeeds(self) -> np.array:
        """
        Generates seeds for the unsupervised region growing algorithm by finding local minima of image intensity.

        Returns:
            seeds (np.array): An array of coordinates representing the seed points.
        """
        img_sample = cv2.imread(self.path, 0) # grayscale image
        img_sample = cv2.GaussianBlur(img_sample, (3,3), cv2.BORDER_DEFAULT) # apply some smoothing
        dist = 10 # initialize minimum distance (no. of pixels) between peaks
        flag = False
        # get local minima of image intensity (by inverting the image and getting its maxima) and use them as seeds
        while flag is False:
            seeds = peak_local_max(img_sample.max() - img_sample, min_distance=dist) 
            # iterate until 40 seeds are filtered out
            if seeds.shape[0]<= 40:
                flag = True
            dist += 10 # increment dist variable to create more distinct peaks
        return seeds # return seed list of coordinates
    
    def getSeeds(self, event, x, y, flags, param):
        """
        Handles mouse events to collect seed points: 
        - If the right button is clicked, it closes all windows.
        - If the left button is clicked, it appends the coordinates of the click to the seeds list.

        Args:
            event (int): The type of mouse event.
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags (int): Any relevant flags passed along with the mouse event.
            param (Any): Any extra parameters passed along with the mouse event.
        """
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.seeds.append([int(y),int(x)])
            
    
    def selectSeeds(self, image) -> list:
        """
        Displays the input image in a new window and sets up a mouse callback to select seeds.
        Returns the list of seeds after the user is done selecting them.

        Args:
            image (np.array): The input image to be displayed.

        Returns:
            list: A list of tuples representing the coordinates of the seeds.
        """
        # Display the image in a new window
        cv2.imshow('Left-hand Click to Select seeds', image)
        cv2.setMouseCallback('Left-hand Click to Select seeds', self.getSeeds)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Return the seeds
        return self.seeds
    
    def applyRegionGrow(self, seeds) -> np.array:
        """
        Applies the region growing segmentation algorithm to the given seeds.

        Args:
            seeds (list): A list of seed coordinates to start the region growing from.

        Returns:
            self.segments (np.array): segmented image - a 3D array representing the segmented regions of the image.
        """
        temp = [] #initialize temp list
        for seed in seeds:
            # add seed to temp list
            temp.append(seed)
            # add neighboring points of the seed to temp list
            temp.extend(self.getNeighbor(*seed))
            
        seeds = temp # update seeds list with final temp list (seeds+neighbors)
        for seed in (seeds): # Region Growing
            # unpack coordinates of seed point
            x0, y0 = seed

            # if seed point is not visited and intensity value is greater than zero at seed point,
            if self.visited[x0, y0] == 0:  
                # increment current region counter
                self.currentRegion += 1
                # mark seed point pixel as visited by assigning it an integer value (self.currentRegion)
                self.visited[x0, y0] = self.currentRegion
                # add pixel coordinates to stack
                self.stack.push((x0,y0))
                
                # perform region-growing process while stack is not empty
                while not self.stack.isEmpty():
                    # get coordinates of popped pixel
                    x, y = self.stack.pop()
                    # perform Breadth-First Search to grow region around seed
                    self.BFS(x, y)
                    # increment iteration counter
                    self.iterations += 1
                
                # check whether region should continue growing or stop
                if(self.VisitedAll()):
                    break
                
                # count number of pixels added to the seed region
                count = np.count_nonzero(self.visited == self.currentRegion)
                # if number of pixels in region is less than 8x8 pixels,
                if(count <  8 * 8 ):
                    # remove region at the current seed point (x0, y0)
                    self.resetRegion()  

        # color each pixel based on its region number
        [self.colorPixel(i,j) for i, j in itertools.product(range(self.h), range(self.w))]
        # return the segmented image
        return self.segments
    
    def distance(self, x, y, x0, y0) -> float:
        """
        Calculates the distance between two pixels. 
        If flag is set, calculates the Euclidean distance (RGB), otherwise calculates the absolute difference (grayscale).

        Args:
            x (int): The x-coordinate of the first pixel.
            y (int): The y-coordinate of the first pixel.
            x0 (int): The x-coordinate of the second pixel.
            y0 (int): The y-coordinate of the second pixel.

        Returns:
            float: The distance between the two pixels.
        """
        if self.flag:
            # calculate Euclidean distance (RGB)
            return np.linalg.norm(self.img[x0, y0] - self.img[x, y])
        else:
            # calculate absolute difference (grayscale)
            return abs(int(self.img[x0, y0]) - int(self.img[x, y]))
    
    def BFS(self, x0, y0):
        """
        Performs Breadth-First Search to expand the region from seed points to neighboring pixels.

        Args:
            x0 (int): The x-coordinate of the seed point.
            y0 (int): The y-coordinate of the seed point.
        """
        # get the region number of the current seed (x0, y0)
        regionNum = self.visited[x0,y0]
        # set the specified homogeneity threshold
        thresh = self.thresh
        # get neighboring points around seed point
        neighbors = self.getNeighbor(x0,y0)

        # for each neighboring point,
        for x,y in neighbors:
            # check if point has not been visited and if the distance between the neighboring point and the seed point is less than current variance
            if self.visited[x,y] == 0 and self.distance(x,y,x0,y0) < thresh:
                
                # check whether region should continue growing or stop
                if(self.VisitedAll()):
                    break
                
                # mark neighboring point as visited by assigning it an integer value (regionNum)
                self.visited[x,y] = regionNum # add point to seed point region
                # add neighboring point to stack
                self.stack.push((x,y))
    
    def VisitedAll(self, max_iteration=2e5) -> bool:
        """
        Checks whether the region growing process should continue or stop (iteration termination condition). 
        The process stops if the number of iterations exceeds a specified maximum or if all pixels in the image have been visited.

        Args:
            max_iteration (int, optional): The maximum number of iterations. Defaults to 2e5.

        Returns:
            bool: True if the region growing process should stop, False otherwise.
        """
        return self.iterations > max_iteration or np.all(self.visited > 0)    
    
    def resetRegion(self):
        """
        Erases the region around the seed point by setting all pixels in the current region to have a label of 0 (unvisited) and decrementing the region counter.
        """
        # set all pixels in the current region (identified by self.currentRegion) to have a label of 0 (unvisited)
        self.visited[self.visited==self.currentRegion] = 0   
        # decrement the region counter
        self.currentRegion -= 1

    def colorPixel(self, i, j):
        """
        Colors a pixel in a region based on its region label. 
        If the region label is 0 (unvisited), the pixel is set to white (RGB value of (255, 255, 255)). 
        Otherwise, the pixel is assigned a color based on the region label:
        - Red component: val * 35
        - Green component: val * 90
        - Blue component: val * 30

        Args:
            i (int): The x-coordinate of the pixel.
            j (int): The y-coordinate of the pixel.
        """
        # get region label for pixel at coordinates (i, j) 
        val = self.visited[i][j]
        # assign color to pixel and store the color in segmented 3D array (image)
        self.segments[i][j] = (255, 255, 255) if (val==0) else (val*35, val*90, val*30)


class Stack():
    """
    Defines a Stack data structure using a list.
    """
    def __init__(self):
        """
        Initializes the Stack with an empty list.
        """
        self.item = []
        
    def push(self, value):
        """
        Adds a value to the top of the Stack (LIFO).

        Args:
            value (Any): The value to be added.
        """
        self.item.append(value)

    def pop(self) -> any:
        """
        Removes and returns the value at the top of the Stack (LIFO).

        Returns:
            Any: The value at the top of the Stack.
        """
        return self.item.pop()

    def size(self) -> int:
        """
        Returns the size of the Stack.

        Returns:
            int: The number of items in the Stack.
        """
        return len(self.item)

    def isEmpty(self) -> bool:
        """
        Checks if the Stack is empty.

        Returns:
            bool: True if the Stack is empty, False otherwise.
        """
        return self.size() == 0