import numpy as np
import pandas as pd
from tqdm import tqdm
import copy as copy
from collections import defaultdict
import operator
import matplotlib.pyplot as plt
from tabulate import tabulate

def pso(f, n, S = 20, lo = -1, up = 1, its = 20, conv_lim = 5, alf = 0.9, cc = 4.25, sc = 1.4):


    
    """
    Input:  
                f: real valued function to minimize, 
                n: the dimension of the search space,
                S: the swarm size, 
                lo: the lower bound of search space, 
                up: the upper bound of search space, 
                its: maximum number of iterations,
                conv_lim: exit if global best is unchanged for conv_lim iterations
                alf: learning rate of velocity,
                cc: cognitive coefficient,
                sc: social coefficient
                                                                     
        Output: 
                found minmum
                number of PSO interations
                number of computational operations
                
                                                    
    """
    
    #initialize particles randomly 
    # a dictionary of particles, keys: 0,...,S-1 and values: random array of size n
    
    parts = defaultdict(lambda: np.random.uniform(lo,up,n))
    
    for i in range(S):
        parts[i] 
        
    #initialize particle velocity
    # a dictionary of particle velocities, keys: 0,...,S-1 and values: random array of size n
    
    ran = (up-lo)/2
    velo = defaultdict(lambda: np.random.uniform(-ran,ran,n))
    
    for i in range(S):
        velo[i]
    
    #initialize current best as initial swarm
    #different objects thanks to deep copy!
    best = copy.deepcopy(parts) 
        
    #initialize current global best randomly
    glob = np.random.uniform(lo,up,n)
    

    # initialize iteration and no global improvment counters 
    count = 0
    conv = 0
    ops = 0

    while count < its and conv <= conv_lim:
        
        count += 1     
        conv += 1
        
        # adapt learning rate, comment it to keep fixed
        # alf = a_min + (a_max - a_min)*count/its

        for p in range(S): #for all particles 

            for i in range(n): #for all dimensions
                
                # total operations counter
                ops += 1

                #initialize weights for deviation from particle/ global best
                wp, wg = np.random.uniform(0,1,2)  

                velo[p][i] = (1-alf) * velo[p][i] + cc*wp * (best[p][i] -  parts[p][i]) + sc*wg * (glob[i] -  parts[p][i])
                
                #update particle position
                parts[p][i] = parts[p][i] + velo[p][i]
                
            #check for and maybe update new best/global position    
            if f(parts[p]) < f(best[p]):
                
                for i in range(n):
                
                    best[p][i] = parts[p][i]
                    
                if f(parts[p]) < f(glob):

                    # global best changed, set convergence counter back to 0
                    conv = 0
                    
                    glob = parts[p]
                            
    return glob, count, ops

def gini(data, coeffs, thr):
    
    n = len(data)
    
    """Inputs:
                data: a 2D array, first column the target of 0/1s
                coeffs: the coefficients of the hyperplane
                thr: the threshold/ RHS of the hyperplane
                                                            
        Output:
                Gini impurity at node
                                                            """
                
    
    # split into target column and feature matrix
    y,X = np.hsplit(data,[1])
    
    
    # evalute points in hyperplane
    plug = np.dot(X,coeffs)
    
    
    # create boolean for above below plane
    # mask_a = (plug > thr)
    mask_a = (plug + thr <= 0)
    mask_b = list(map(operator.not_, mask_a))
    
    
    # get the partition
    ab = y[mask_a]
    bel = y[mask_b]
    
    
    # get frequencies of class y=1 in both partitions
    p_a = 0
    p_b = 0
    
    if len(ab) != 0:
        p_a = sum(ab)/len(ab)
        
    if len(bel) != 0:
        p_b = sum(bel)/len(bel)
        
    
    # calc Gini impurity at node as weighted avergage of Gini impurities in partitions
    return len(ab)/n * 2*p_a*(1-p_a) + len(bel)/n * 2*p_b*(1-p_b) 

def gen(data):
    """Input:
            data: array like, first column the 0/1 target
            
        Output:
            f: function, takes in hyperplane coefficients and returns Gini impurity for data under the given split"""
    
    def f(params):
            
        thr = params[-1]
        coeffs = params[:-1]
            
        return gini(data, coeffs, thr)
    
    return f

def gen_uni(data, feature, d):
    """Input
            data: array like, first column the 0/1 target
            feature: column index in feature matrix of feature to split on 
            d: number of features
            
        Output
            f: function, takes in hyperplane coefficient and returns Gini impurity for data under the given split"""
    
    def f(params):

        thr = params[1]

        # set all coefficents to 0, except of feature you want to split on   
        coeffs = np.zeros(d)
        coeffs[feature] = params[0]
            
        return gini(data, coeffs, thr)
    
    return f

def line(params, ran = [-3,3]):

    """Used in plot method of tree object"""

    if params[1] != 0:
        return [(p,(-params[2]-params[0]*p)/params[1]) for p in ran]
    
    else:
        return [(-params[2]/params[0],p) for p in ran]

def minmax(data):
    """
    Scales columns of array
    
    """

    def h(col):
    
        col_min = np.min(col)
        col_max = np.max(col)

        if (col_max - col_min) == 0:
            return 0
        else:
            return (col - col_min)/(col_max - col_min)
    
    y,X = np.hsplit(data,[1])
    
    X = np.apply_along_axis(h, 0, X)

    return np.column_stack((y,X))

def balance(data):

    """
    Returns proportions of observations by class
    Classes need to be 1, 2, 3, ...
    """

    y,X = np.hsplit(data,[1])

    bins = y.flatten()
    bins = bins.astype(int)

    bins = np.bincount(bins)

    print(bins[1:]/len(y))

class node:

    """Attributes:
                    type: string, is root, internal or leaf 
                    name: string, encodes position in tree with sequence of 0/1s, e.g.
                                          x                                         
                                         / \ 
                                       x0   x1  
                                           /  \       
                                        x10   x11  
                                             /   \ 
                                           x110  x111     
                    split: 1D array, hyperplane coeffiecents, e.g. [a, b, c] for ax + by + c = 0, for root/internal
                    label: string, is red or blue, for leaf
                    impurity: Gini impurity under training data
                    size: number of training examples at node
                    data: the training examples at the node
                    prob: float, assigned probability of beloning to class 1
                    its: int, the number of PSO iterations that produced the split
                    feature: column index of feature considered in split (axis-parallel tree)
                    before: for leaves, True if node became leaf before trying to split it based on critera available at that point (see tree.grow())
                    """

    def __init__(self, type, name, split):
        
        self.type = type 
        self.name = name
        self.split = split
        self.label = None
        self.impurity = None
        self.size = None
        self.data = None
        self.prob = None
        self.its = None
        self.feature = None
    
    def eval(self, x):

        """Evaluate the position of an example relative to the splitting hyperplane at this node.
           Returns "0" if above and "1" if below.
           Used in classify method of tree object or for single classification."""
        
        # plug example into hyperplane
        V = np.dot(self.split[:-1],x) + self.split[-1]

        # go right or left in tree
        if V > 0: return "0" # left
        else: return "1"     # right

    def inspect(self):

        """Get some relevant information on the node."""

        if self.type == "leaf":
            print(f"name: {self.name}, type: {self.type}, label: {self.label}, assigned probability: {round(self.prob[0], 2)}, impurity: {round(self.impurity[0], 2)}, size: {self.size}, before: {self.before} ")

        if self.type != "leaf":
            print(f"name: {self.name}, type: {self.type}, split: {self.split}, impurity: {round(self.impurity[0], 2)}, size: {self.size}, PSO-iterations: {self.its}")

    def plot(self):

        """
        Plots all training examples at node and split if note is root/internal.
        For binary classes only
        
        """


        # prepare and plot data
        y,X = np.hsplit(self.data,[1])
        y = np.dot(y,[1])

        # find out number of classes
        bins = y.flatten()
        bins = bins.astype(int)

        bins = np.bincount(bins)

        maximum = max(bins)

        for i in range((maximum+1)):
            globals()[f'mask{i}'] = (y == (i+1))

            globals()[f'X{i}1'] = X[globals()[f'mask{i}'],0]
            globals()[f'X{i}2'] = X[globals()[f'mask{i}'],1]
        
        
        # set limits
        l0 = min(X[:,0])-0.25
        u0 = max(X[:,0])+0.25
        l1 = min(X[:,1])-0.25
        u1 = max(X[:,1])+0.25

        fig, ax = plt.subplots()

        plt.xlim([l0, u0])
        plt.ylim([l1, u1])

        # plot data at node
        for i in range((maximum+1)):

            try:
                plt.plot(globals()[f'X{i}1'], globals()[f'X{i}2'], "black",linestyle='', marker = f"${i+1}$", mfc = "black")
            
            except: None
        
        # optional: plot splitting lines

        if self.type == "internal":

            ax.axline(*line(self.split ))

class tree:
    
    def __init__(self, data, K, lab = False):
        
        """
        Inputs
            data: array, first column the target encoded in 0/1, other columns the features
            lab: boolean, should data be split into test and training set?
            K: int, the number of classes
        
        Attributes
            data_train: array, first column the target encoded in 0/1
            data_test: array, test set is available if lab == True
            nodes: dictionary of node objects, keys are node names e.g. x01, values are node objects
            nr_splits: int, the number of splits in the tree including the root splits, available once tree is grown
            op: list, length is number of times PSO was run while growing the tree, 
                      elemets are sum of steps of all particle elements (interesting for run time)
            
                      
        """
        
        self.nodes = {} 
        self.nr_splits = 0
        self.oblique = None
        self.K = K

        self.op = []

        # split data into training and test set
        if lab == True:

            # set proportion of data to use in training 
            alf = 0.75

            N = len(data)
            n = int(alf*N)

            # randomize 
            np.random.shuffle(data)

            # split 
            self.data_train = data[:n]
            self.data_test = data[n:]

        # use entire data as training set
        else:
            self.data_train = data
            self.data_test = None


    def grow(self, move = "", data = None, hist = "x", S = 20, max_depth = None, imbal = 0.5, lim = 1, min_size = None, oblique = True):
        
        """Recurisvely splits the data using Gini impurity and PSO.
           Creates nodes and assigns them either the split (root/internal) or a label (leaf)
           Nodes are stored in the tree attribute nodes, a dictionary.
        
        Inputs:
            move: string, 0/1 determines whether to move left or right in tree
            data: array, the data set at the given node
            hist: string, parent node label
            max_depth: int, limits depth of tree (a tree of depth 2 has maximally four leafs)
            imbal: float, threshold for the classification of imbalanced data
            lim: int, data at node will not be split further if there are less than lim examples
            min_size: int, smallest possible leaf"""

        # initialize data
        if np.all(data) == None:
            data = self.data_train

        # initialize maxiumum depth
        if max_depth == None:
            max_depth = float("inf")


        # initialize minimum leaf size to fraction of training set
        if min_size == None:
            frac = 0.01
            min_size = round(frac*len(self.data_train))

        # own name is parnent name plus own move
        call = hist + move

        if call == "x":
            self.nr_splits = 0
            self.op = []
            self.oblique = oblique # not sure what this does?

        # split data into target and feature matrix
        y,X = np.hsplit(data,[1]) 

        
        # exit if maximum depth is reached or node is pure or too little observations
        if (len(call) > max_depth) or (np.all(y == y[0])) or len(y) < lim :

            # determine leaf label and save as string

            bins = y.flatten()
            bins = bins.astype(int)

            bins = np.bincount(bins)
            maximum = max(bins)

            maj_cl_l = []

            for i in range(len(bins)):
                if bins[i] == maximum:
                    maj_cl_l.append(i)

            # sample uniform randomly from list of joint majority classes
            maj_cl = np.random.choice(maj_cl_l)
            label = str(maj_cl)

            # create leaf
            self.nodes[call] = node("leaf", call, None)
            self.nodes[call].label = label
            self.nodes[call].size = len(y) 
            self.nodes[call].data = data
            self.nodes[call].before = True

            # calcualte and assign fittet probability
            maj_frequ = bins[maj_cl]

            self.nodes[call].prob = maj_frequ/len(y)

            # assign impurity from majority class perspectivc
            self.nodes[call].impurity = 2 * self.nodes[call].prob * (1 - self.nodes[call].prob)

            #exit
            return None
        
        # create list of impurites by class under consideration
        imps = {}
        pars = {} # keys are classes 1,2,3,4,... values are paramters of hyperplane

        # check which classes are present at this node
        bins = y.flatten()
        bins = bins.astype(int)
        bins = np.bincount(bins)

        # for all K classes
        for k in range(len(bins)):
            
            if bins[k] != 0:

                # initialize binary OvR target
                y_bin = np.zeros(len(y))

                # convert to binary
                for obs in range(len(y)):
                    if y[obs] == k:
                        y_bin[obs] = 1
                
                # binary data set
                data_bin = np.column_stack((y_bin, X))
            
                if oblique:
                    # create objective function
                    f = gen(data_bin)

                    # find best split
                    n = data_bin.shape[1]
                    par, its, ops = pso(f, n, S)
                    c = par[-1]
                    coef = par[:-1]

                    # count PSO operations
                    self.op.append(ops)

                    # add impurity to list and save found paramters
                    imps[k] = gini(data_bin, coef, c)
                    pars[k] = par
                
                else:
                    # get the number of features
                    d = self.data_train.shape[1] - 1

                    # create lists of impurities and coeffcients to compare
                    imp = []
                    p = []

                    # for all features, check the impurity of the split
                    for i in range(d):

                        f = gen_uni(data_bin, i, d)

                        # dimension of search space is 2
                        x_, its, ops = pso(f, 2, S)

                        # create coefficent vector
                        c = x_[1]
                        coef = np.zeros(d)
                        coef[i] = x_[0]

                        self.op.append(ops)

                        # calculate and save impurity and coefficients under this split
                        imp.append(gini(data_bin, coef, c))
                        p.append([coef, c, its])
                    
                    # find feature with best split and get coeffs
                    feature = imp.index(min(imp))
                    coef = p[feature][0]
                    c = p[feature][1]
                    its = p[feature][2]

                    # join hyperplane coefficients
                    par = np.append(coef, c)

                    # add impurity to list
                    imps[k] = gini(data_bin, coef, c)
                    pars[k] = par


        # check which class to use as reference in this split and get associated paramters
        klas = min(imps)
        par = pars[klas]
        c = par[-1]
        coef = par[:-1]

        # create subsets
        v = np.dot(X,coef)
        
        # create boolean for above below plane
        mask_a = (v + c > 0)
        mask_b = list(map(operator.not_, mask_a))

        # get the partition
        a = data[mask_a,]
        b = data[mask_b,]

        # if split produces children with too little observations, parent should become leaf
        if len(a) < min_size or len(b) < min_size:
            
            self.nodes[call] = node("leaf", call, None)

            # determine leaf label, depending on imbalance threshold

            bins = y.flatten()
            bins = bins.astype(int)

            bins = np.bincount(bins)
            maximum = max(bins)

            maj_cl_l = []

            for i in range(len(bins)):
                if bins[i] == maximum:
                    maj_cl_l.append(i)

            # sample uniform randomly from list of joint majority classes
            maj_cl = np.random.choice(maj_cl_l)
            label = str(maj_cl)

            self.nodes[call].label = label
            self.nodes[call].size = len(y) 
            self.nodes[call].data = data
            self.nodes[call].before = False

            # calcualte and assign fittet probability
            maj_frequ = bins[maj_cl]

            self.nodes[call].prob = maj_frequ/len(y)

            # assign impurity from majority class perspectivc
            self.nodes[call].impurity = 2 * self.nodes[call].prob * (1 - self.nodes[call].prob)

            # exit
            return None

        # create internal/root node    
        self.nodes[call] = node("internal", call, par)
        self.nodes[call].data = data
        self.nodes[call].impurity = imps[min(imps)] # impurity achieved in binary minimzation problem
        self.nodes[call].size = len(y)
        self.nodes[call].its = its

        # increase number of splits
        self.nr_splits += 1

        # recursion step
        self.grow("0", a, call, S, max_depth, imbal, lim, min_size, oblique) # move above split <=> left in tree <=> "0"
        self.grow("1", b, call, S, max_depth, imbal, lim, min_size, oblique) # move below split <=> right in tree <=> "1"


    def plot(self, dots = True, shade = True, lines = False, test = False):
        
        """ For 2D features only!
            Plots all training data (dots) and shades area according to label assigned (shade) by default.
            Plot splits if option lines == True.
            Only call on grown tree.

            For binary classes only.
        """

        # prepare and plot data
        if test == False:
            y,X = np.hsplit(self.data_train,[1])

        else:
            y,X = np.hsplit(self.data_test,[1])

        y = np.dot(y,[1])

        for i in range(self.K):
            globals()[f'mask{i}'] = (y == (i+1))

            globals()[f'X{i}1'] = X[globals()[f'mask{i}'],0]
            globals()[f'X{i}2'] = X[globals()[f'mask{i}'],1]
        
        
        fig, ax = plt.subplots()

        # set limits
        l0 = min(X[:,0])-0.25
        u0 = max(X[:,0])+0.25
        l1 = min(X[:,1])-0.25
        u1 = max(X[:,1])+0.25

        plt.xlim([l0, u0])
        plt.ylim([l1, u1])

        # plot training data
        if dots:
            for i in range(self.K):
                plt.plot(globals()[f'X{i}1'], globals()[f'X{i}2'], "black",linestyle='', marker = f"${i+1}$", mfc = "black")
            
        
        # optional: plot splitting lines
        if lines:
            count = 0
            for key,value in self.nodes.items():
                    
                    # check if node is internal
                    if value.type == "internal":
                        count += 1

                        ax.axline(*line(value.split), linestyle = (0, (count, 2*(count-1))), label = key)
                        
            plt.legend()

        # create grid for shading
        x0 = np.linspace(l0,u0,2000)
        x1 = np.linspace(l1,u1,2000)

        x0,x1 = np.meshgrid(x0,x1) 

        # start counting leafs
        leaf_nr = 0

        # default: shade areas according to assigned label
        if shade:
            
            # walk through all nodes
            for key, value in self.nodes.items():
                
                # consider the leafs only
                if (value.type == "leaf") & (value.label == "blue"):

                    leaf_nr += 1
                    name = value.name

                    # start at root
                    # create function from split
                    a,b,c = self.nodes[name[0]].split
                    globals()[f'f_{1}'] = lambda x0,x1 : a*x0 + b*x1 + c
                    
                    # create linear restriction depending on next step in tree
                    if name[1] == "0":
                        globals()[f'lin_res{leaf_nr}'] = (globals()[f'f_{1}'](x0,x1) > 0)

                    else:
                        globals()[f'lin_res{leaf_nr}'] = (globals()[f'f_{1}'](x0,x1) <= 0)

                    # walk through all splits that lead to this node
                    for i in range(2,len(name)):
                        
                        # create function from split
                        a,b,c = self.nodes[name[:i]].split
                        globals()[f'f_{i}'] = lambda x0,x1 : a*x0 + b*x1 + c

                        # add restriction to set of restrictions depending on next step in tree
                        if name[i] == "0":
                            globals()[f'lin_res{leaf_nr}'] = globals()[f'lin_res{leaf_nr}'] & (globals()[f'f_{i}'](x0,x1) > 0)

                        else:
                            globals()[f'lin_res{leaf_nr}'] = globals()[f'lin_res{leaf_nr}'] & (globals()[f'f_{i}'](x0,x1) <= 0)
                
            # first element of chain of OR statements
            res = globals()[f'lin_res{1}']
                
            # put all the leaves together in chain of OR statements
            for i in range(2,(leaf_nr+1)):

                res = res | globals()[f'lin_res{i}']   

            # shade based on restrictions 
            im = plt.imshow( (res).astype(int) , 
                    extent=(x0.min(),x0.max(),x1.min(),x1.max()),
                    origin="lower",
                    cmap="bwr_r") 

        if self.oblique:
            plt.title("Oblique Splits")
        else:
            plt.title("Standard Splits")

        # plot
        plt.show()


    def splits(self):

        """Lists all splits and their hyperplane coefficients."""

        for key,value in self.nodes.items():

            # check if node is internal or root
            if np.all(value.split) != None:
                print(f"Split at node {key}:{value.split}")


    def leaves(self):

        """Lists all the leaves, their label, their training impurity and training size."""

        for key,value in self.nodes.items():

            # check if node is a leaf
            if value.type == "leaf":
                print(f"Label at leaf node {key}:{value.label}, impurity {round(value.impurity[0], ndigits = 2)}, size {value.size}")


    def classify(self, x):

        """Classifies an example x

           returns string label 1, 2, 3, 4, ..."""

        # start at the root
        Node = self.nodes["x"]

        # walk through the tree until you reach a leaf
        while Node.type != "leaf":

            name = Node.name

            # check whether to move above or below hyperplane
            move = Node.eval(x)

            # move to next node
            Node = self.nodes[name+move]
        
        lab = Node.label

        return lab
            
    
    def test(self, test_data = None):

        """Test and report results
           Use either the test data given at initialization or supply new test_data"""

        # make supplied test set available as attribute of tree
        if np.all(test_data) != None:
            self.data_test = test_data

        # split and coerce 
        y_test, X_test = np.hsplit(self.data_test,[1])
        y_test = np.dot(y_test,[1])

        # classify test data
        y_fit = np.apply_along_axis(self.classify, 1, X_test)

        # get count of classes in training data
        bins = y_test.flatten()
        bins = bins.astype(int)
        bins = np.bincount(bins)

        # initialize list of K accuracies
        acc = [0 for i in range(self.K)]

        for i in range(len(y_test)):

            if y_test[i] == int(y_fit[i]):
                acc[int(y_test[i])-1] += 1
        
        for i in range(self.K):
            acc[i] = acc[i]/bins[i+1]

        return list(np.round(acc,2))   

class forest:

    def __init__(self, data, K,lab = True):
        
        """
        Inputs
            data: array, first column the target encoded in 0/1, other columns the features
            lab: boolean, should data be split into test and training set?
            K: int, the number of classes
        
        Attributes
            data_train: array, first column the target encoded in 0/1
            data_test: array, test set is available if lab == True
            size: int, the number of trees
            trees: dict, keys are integers 0, ..., size-1, values are tree objects
            boots: dict, keys are integers 0, ..., size-1, values are the resampled data sets the trees are grown on
            feature sets: dict, the keys are integers 0, ..., size-1, values are column indices
                          referring to features considered by tree"""
        
        # split data into training and test set
        if lab:

            # set proportion of data to use in training 
            alf = 0.75

            N = len(data)
            n = int(alf*N)

            # randomize 
            np.random.shuffle(data)

            # split 
            self.data_train = data[:n]
            self.data_test = data[n:]

        # use entire data as training set (with intention to later provide a test set in test method)
        else:
            self.data_train = data
            self.data_test = None

        # initialize
        self.size = 0
        self.trees = {}
        self.boots = {}
        self.feature_sets = {}
        self.avg_nr_splits = 0
        self.K = K

    def grow(self, size = 10, m = None, max_depth = None, imbal = 0.5, lim = 30, min_size = None, oblique = True, S = 20):

        """
        Create tree objects grown on bootstrapped data sets and store them in class attribute self.trees.

        Inputs 
                size: int, number of trees to grow
                m: int, number of features to consider per tree, by default the square root of available
                max_depth: int, maximum depth of trees 
                imbal: float, the threshold for imbalanced data
                lim: int, data at node will not be split further if there are less than lim examples
                min_size: int, smallest possible leaf
                oblique: bool, grow oblique forest """
        
        # initialize

        if max_depth == None:
            max_depth = float("inf")

        self.size = size

        nr_splits = []

        # initialize minimum leaf size to fraction of training set
        frac = 0.01
        if min_size == None:
            min_size = round(frac*len(self.data_train))

        # get the number of features
        d = self.data_train.shape[1]-1

        if m == None:
        
            # get number of features used in tree
            m = round(np.sqrt(d))
        
        N = self.data_train.shape[0]

        # for all trees generate boot strapped data
        for i in range(self.size):

            # create radnom sample from selected features
            #  of same size as original data, with replacement
            rows = np.random.choice(N, size = N, replace = True)
            cols = np.random.choice(range(1,d+1), size = m, replace = False)

            #order the chosen features
            cols = np.sort(cols)

            # save the features that tree i is trained on
            self.feature_sets[i] = cols

            # add target (col index 0) back
            cols = np.insert(cols,0,0)

            # create training set for tree i (start with first row)
            globals()[f"dat_{i}"] = np.array(self.data_train[rows[0],:])

            # add the other N-1 rows
            for j in range(1,N):

                globals()[f"dat_{i}"] = np.row_stack((globals()[f"dat_{i}"], np.array(self.data_train[rows[j],:])))

            # retain only randomly selected features and target, as listed in cols
            globals()[f"dat_{i}"] = globals()[f"dat_{i}"][:,cols]

            # save i-th bootstrapped sample
            self.boots[i] = globals()[f"dat_{i}"]

            # initialize and grow tree on bootstrapped dataset
            globals()[f"T_{i}"] = tree(globals()[f"dat_{i}"], K = self.K, lab = False)

            globals()[f"T_{i}"].grow(S = S, max_depth = max_depth, lim = lim, min_size = min_size, oblique = oblique)

            # info thats nice to have
            nr_splits.append(globals()[f"T_{i}"].nr_splits)

            # save tree
            self.trees[i] = globals()[f"T_{i}"]
        
        # info thats nice to have
        self.avg_nr_splits = sum(nr_splits)/self.size

    def classify(self, x):

        """
        Input
            x: array, an example to be classified 

        Output
            class label: int, 0/1
        """
        # initialize list of votes, each tree has a vote
        votes = []

        # for all trees make a classification decision
        for i in range(self.size):

            # get the features that tree i is grown on
            x_i = x[self.feature_sets[i]]

            votes.append( self.trees[i].classify(x_i) )

        # classify based on majority vote, note the slight bias towards 1 in an sample of even size
        bins = np.bincount(votes)
        maximum = max(bins)

        maj_vot = []

        for i in range(len(bins)):
            if bins[i] == maximum:
                maj_vot.append(i)

        return np.random.choice(maj_vot)
    
    def plot(self):

        """ For 2D features only!
            Plots all training data.

            For binary target.
            """

        # prepare and plot training data
        y,X = np.hsplit(self.data_train,[1])
        y = np.dot(y,[1])
        
            
        mask1 = (y == 1)
        mask0 = (y == 0)
        
        X11 = X[mask1,0]
        X12 = X[mask1,1]
        X01 = X[mask0,0]
        X02 = X[mask0,1]
        
        fig, ax = plt.subplots()

        # set limits
        l0 = min(X[:,0])-0.25
        u0 = max(X[:,0])+0.25
        l1 = min(X[:,1])-0.25
        u1 = max(X[:,1])+0.25

        plt.xlim([l0, u0])
        plt.ylim([l1, u1])

        # plot training data
        plt.plot(X11,X12, "black",linestyle='', marker = "o",mfc = "red")
        plt.plot(X01,X02, "black", linestyle='', marker = "s", mfc = "blue")
    
    def test(self, test_data = None):

        """
        Test and report results

        Input
       
           test_data: array, if lab == False at initialization or one wants to test on different data
           
        Output
               accuries by class
               
        """

        # make supplied test set available as attribute of forest
        if np.all(test_data) != None:
            self.data_test = test_data

        # split and coerce 
        y_test, X_test = np.hsplit(self.data_test,[1])
        y_test = np.dot(y_test,[1])

        # classify test data
        y_fit = np.apply_along_axis(self.classify, 1, X_test)

        # get count of classes in test data
        bins = y_test.flatten()
        bins = bins.astype(int)

        # first bin will always be empty cause classes are 1, 2, 3, ...
        bins = np.bincount(bins)

        # initialize list of accuracies
        acc = [0 for i in range(self.K)]

        for i in range(len(y_test)):

            if y_test[i] == int(y_fit[i]):

                # add successful prediction to correct bin bin
                acc[int(y_test[i])-1] += 1
        
        for i in range(self.K):

            # start from second bin 
            acc[i] = acc[i]/bins[i+1]

        return list(np.round(acc,4))
    
def tree_sim(data, K, N, max_depth, lim, min_size, oblique, S, test = None):
    """
    Does N repetitions of growing and then testing tree.
    Test either on supplied test set or in part of data.

    Output: average accuracy for every class
    
    """
    # check if test set was supplied
    if type(test) is np.ndarray:
        flag = False
    else:
        flag = True

    # grow and test first tree outside of loop (cause you need existing array to append rows to it)
    T = tree(data, K = K, lab = flag)
    T.grow( max_depth = max_depth, lim = lim, min_size = min_size, oblique = oblique, S = S)
    a = T.test(test_data = test)

    acc = np.array(a)

    # do the other N-1 trials
    for i in tqdm(range(N-1)):

        T = tree(data, K = K,lab = flag)
        T.grow(max_depth = max_depth, lim = lim, min_size = min_size, oblique = oblique, S = S)
        a = np.array(T.test(test_data = test))

        acc = np.row_stack((acc,a))

    
    return list(np.round(np.mean(acc, axis = 0),4))

def forest_sim(data, K, N, size, max_depth, lim, min_size, oblique, S, m = None, test = None): 

    """
    Does N repetitions of growing and then testing forest.
    Test either on supplied test set or in part of data.

    Output: average accuracy for every class
    
    """
    # check if test set was supplied
    if type(test) is np.ndarray:
        flag = False
    else:
        flag = True

    # grow and test first forest outside of loop (cause you need existing array to append rows to it)
    F = forest(data, K, lab = flag)
    F.grow(size = size, m = m, max_depth = max_depth, lim = lim, min_size = min_size, oblique = oblique, S = S)
    a = F.test(test_data = test)

    acc = np.array(a)

    # do the other N-1 trials
    for i in tqdm(range(N-1)):

        F = forest(data, K = K, lab = flag)
        F.grow(size = size, m = m, max_depth = max_depth, lim = lim, min_size = min_size, oblique = oblique, S = S)
        a = np.array(F.test(test_data = test))

        acc = np.row_stack((acc,a))

    
    return list(np.round(np.mean(acc, axis = 0),4))
