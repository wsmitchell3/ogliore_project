import numpy as np
from mswd import *

def run_filament(cycles=80, Y_time=3, X_time=3, Y=50000, X=50000,\
                 other_time=0, beam_decay_pct_min=0, return_counts=False):
    """Simulate averaging ratios (and total counts) over the course of one
    filament.

    Returns a list of four or six items (names here are from Ogliore NIMB 2011 https://arxiv.org/abs/1106.0797):
    [0] -- r2: Mean of the ratios
    [1] -- Relative deviation from the true mean
    [2] -- r1: Ratio based on entire summed counts
    [3] -- Relative deviation from the true mean
    [4] -- Sum Y counts (if return_counts is True)
    [5] -- Sum X counts (if return_counts is True)
    [6] -- 2-sigma standard error of r2

    Keyword arguments:
    cycles -- the number of ratios to measure (default 80)
    Y_time -- the number of seconds to measure numerator species Y (default 3)
    X_time -- the number of seconds to measure denominator species X (default 3)
    Y -- the true abundance of species Y in counts per second (default 50,000)
    X -- the true abundance of species X in counts per second (default 50,000)
    other_time -- the time each cycle not spent measuring X or Y, such as
        when measuring other isotopes (default 0)
    beam_decay_pct_min -- the beam decay rate in percent/minute (default 0)
    return_counts -- pass the Y and X count arrays back if this is True
    
    """
    # Convert everything to doubles; the original values may be integers, but
    # the ratios can take on non-integral values
    X_time=np.double(X_time)
    Y_time=np.double(Y_time)
    other_time=np.double(other_time)
    beam_decay_pct_min=np.double(beam_decay_pct_min)
    Y=np.double(Y)
    X=np.double(X)
    R=Y/X # The true ratio

    # Turn the beam decay rate into a decay constant with units of per second
    decay_const=-np.log(1-beam_decay_pct_min/100/60)

    # Create arrays of the proper type to hold counts and ratios
    x=np.double(range(cycles))
    y=np.double(range(cycles))
    r2=np.double(range(cycles))

    # Set the time to zero
    time_elapsed=np.double(0)

    # Calculate the total time spent in each cycle
    cycle_time=np.double(other_time+X_time+Y_time)

    # Make some measurements
    if (beam_decay_pct_min == 0): # If decay rate is 0, don't calculate decay
        for i in range(cycles):
            #print exp(-decay_const*time_elapsed)
            #print time_elapsed
            # Measure species X with a poisson distribution for X_time seconds
            # Beam intensity changes as an exponential (decay)
            # Result is in counts per second
            x[i]=0
            x_tries=0
            while (x[i]==0):
                #x[i]=np.double(np.random.normal(X_time*X, np.sqrt(X_time*X)))
                x[i]=np.double(np.random.poisson(X_time*X))
                if (x_tries > 10 and x[i]==0):
                    x[i] = 1

            # Measure species Y with a poisson distribution for X_time seconds
            y[i]=0
            y_tries = 0 
            while (y[i]==0):
                #y[i]=np.double(np.random.normal(Y_time*Y, np.sqrt(Y_time*Y)))
                y[i]=np.double(np.random.poisson(Y_time*Y))
                if (y_tries > 10 and y[i]==0):
                    y[i] = 1
            # Calculate the ratio of y/x from this measurement
            r2[i]=y[i]/x[i]*X_time/Y_time
                
    else:  # Beam decay rate is non-zero; include decay correction
        for i in range(cycles):
            x[i]=0
            while (x[i]==0):
                x[i]=np.double(np.random.poisson(X_time*X\
                                             *np.exp(-decay_const*time_elapsed)))

            # Measure species Y with a poisson distribution for X_time seconds
            y[i]=0
            while (y[i]==0):
                y[i]=np.double(np.random.poisson(Y_time*Y\
                                             *np.exp(-decay_const*time_elapsed)))

            # Calculate the ratio of y/x from this measurement
            r2[i]=y[i]/x[i]*X_time/Y_time
            time_elapsed += cycle_time  # Increment the elapsed time
    r1 = X_time/Y_time * sum(y)/sum(x)  # Calculate the average based on total counts
    #print("r2: " + str(mean(r2))+ "\t" + str((mean(r2)-R)/R*100) + "%")
    #print("r1: " + str(r1)+ "\t" + str((mean(r1)-R)/R*100) + "%")
    a = np.double(np.mean(r2))  # Take the average of the individual ratios

    # Get the 2-sigma standard error
    a_std = np.double(np.std(r2))/np.sqrt(cycles)*2

    if return_counts:
        return [a, deviation(a,R),r1,deviation(r1,R),sum(y),sum(x),a_std]
    # return the average of ratios, its deviation from the true ratio,
    #   the average from total counts, and its deviation from the true ratio
    return [a, deviation(a,R),r1,deviation(r1,R)]

def run_blocks(blocks=6,cycles=10,Y_time=3,X_time=3,Y=50000,X=50000,\
               block_time=300,beam_decay_pct_min=0):
    """Simulate a block-by-block analysis of a filament

    Returns a list of [blocks] elements, each a list of four items:
    [0] -- Average of the measured ratios (for the block)
    [1] -- Deviation of the average of the ratios (relative)
    [2] -- Ratio based on summed counts (for the block)
    [3] -- Deviation of the summed count ratio (relative)

    Keyword arguments:
    blocks -- Number of blocks to run
    cycles -- Number of measurements within a block
    Y_time -- Time spent measuring isotope Y (seconds)
    X_time -- Time spent measuring isotope X (seconds)
    Y -- Count rate of Y (counts/second)
    X -- Count rate of X (counts/second)
    block_time -- Time for each block (seconds)
    beam_decay_pct_min -- Beam decay rate (percent/minute)
    
    """

    # Cast everything as a double
    X_time=np.double(X_time)
    Y_time=np.double(Y_time)
    block_time=np.double(block_time)
    beam_decay_pct_min=np.double(beam_decay_pct_min)
    Y=np.double(Y)
    X=np.double(X)
    time_elapsed=np.double(0)

    # Calculate time spent per cycle not measuring X or Y
    other_time=block_time/cycles-Y_time-X_time

    # Calculate the decay constant
    decay_const=-np.log(1-beam_decay_pct_min/100/60)

    # Initialize the list to return
    block_stats = range(blocks)

    # Make the measurements
    for i in range(blocks):
        # Run each block by itself
        block_stats[i] = run_filament(cycles,Y_time,X_time,\
                Y*np.exp(-decay_const*time_elapsed),\
                X*np.exp(-decay_const*time_elapsed),\
                other_time,beam_decay_pct_min, True)
        #Increment the time
        time_elapsed += block_time
    return np.array(block_stats)

def analyze_blocks(blocks=6, cycles=10, Y_time=3, X_time=3, Y=50000, X=50000,\
                   block_time=300, beam_decay_pct_min=0, print_blocks=False):
    """Perform block-by-block weighted mean, standard deviation, and MSWD
analysis of a filament

    Returns a list of three items:
    [0] -- The average of the ratios from the entire filament (unweighted)
    [1] -- The ratio of the summed counts from the entire filament
    [2] -- The weighted average of the block-by-block average of ratios

    Keyword args:
    blocks -- Number of blocks to run
    cycles -- Number of measurements within a block
    Y_time -- Time spent measuring isotope Y (seconds)
    X_time -- Time spent measuring isotope X (seconds)
    Y -- Count rate of Y (counts/second)
    X -- Count rate of X (counts/second)
    block_time -- Time for each block (seconds)
    beam_decay_pct_min -- Beam decay rate (percent/minute)
    print_blocks -- Show block-by-block averages if True (default: False)
    
"""

    # Cast everything as a double to avoid integer-math stupidity
    X_time=np.double(X_time)
    Y_time=np.double(Y_time)
    X=np.double(X)
    Y=np.double(Y)

    # Calculate the true ratio
    R=Y/X

    # Get the measurements
    block_stats = run_blocks(blocks, cycles, Y_time, X_time, Y, X,\
                             block_time, beam_decay_pct_min)
    """
    # Initialize arrays to hold the blockwise data
    totals = np.double([0,0,0,0])  # each return value from the block
    block_average=np.double(range(len(block_stats))) # each block average
    block_err=np.double(range(len(block_stats)))  # each block standard deviation
    if print_blocks:
        print "Block statistics"

    # Pull the values out of the nested arrays
    for i in range(len(block_stats)):
        block_average[i]=block_stats[i][0]
        if print_blocks:
            print 'Block {0}: {1:.6f} {2:.6f} {3:.6f} {4:.6f}'.format(i,\
                block_stats[i][0],block_stats[i][1],\
                block_stats[i][2],block_stats[i][3])
        for j in range(4):
            totals[j] = totals[j] + block_stats[i][j]

    # Average the blockwise data together, unweighted
    totals = totals/len(block_stats)
    if print_blocks:
        print '*---------------------------*'
        print 'Averages: {0:.6f} {1:.6f} {2:.6f} {3:.6f}'.format(totals[0],\
                    totals[1], totals[2], totals[3])

    # Initialize arrays for individual measurements
    y = []
    x = []
    sum_x = 0
    sum_y = 0

    # Pull each measurement out of the original data
    for i in range(len(block_stats)):
        for j in range(len(block_stats[i][4])):
            y.append(block_stats[i][4][j])
            x.append(block_stats[i][5][j])

    # Calculate the individually measured ratios, and sum the total counts
    r2 = range(len(y))
    for i in range(len(y)):
        r2[i] = np.double(y[i])/x[i]*X_time/Y_time
        sum_x += x[i]
        sum_y += y[i]

    cycles = len(y)/len(block_stats)  # Determine the number of cycles per block
    for i in range(len(block_stats)):
        # Get the standard deviation in ratios for each block
        block_err[i] = np.std(r2[i*cycles:(i+1)*cycles])
    """
    sum_y = sum(block_stats[:,4])
    sum_x = sum(block_stats[:,5])

    
    a = np.mean(block_stats[:,0]) # Average the ratios together (unweighted)
    b = deviation(a, R)  # Calculate the deviation from the true value
    c = sum_y/sum_x*X_time/Y_time  # Calculate the ratio of summed counts
    d = deviation(c, R)  # Calculate the deviation from the true value

    # Calculate weighted average, standard deviation, and MSWD
    temp_mswd = get_mswd(block_stats[:,0], block_stats[:,6])

    #Print, if specified
    if print_blocks:
        print '*---------------------------*'
        print 'Filament: {0:.6f} {1:.6f} {2:.6f} {3:.6f}'.format(a,b,c,d)

        print '*---------------------------*'
        print "Wtd. Mean\tStd. Dev.\tMSWD"
        print(temp_mswd)

    return [a,c,temp_mswd[0]]
    
def deviation(sample, target):
    """Calculate the relative difference between [sample] and [target]"""
    return np.double(sample-target)/target

def virtual_turret(N=10000, blocks=6, cycles=10, Y_time=3, X_time=3,\
                   Y=50000, X=50000, block_time=300, beam_decay_pct_min=0.0):
    """Acquire statistics on a large number of virtual filament runs,
including block-by-block weighted averages

    Returns a list of three values:
    [0] -- Mean ratio based on averaging ratios individually for each filament (r2)
    [1] -- Mean ratio based on averaging ratios from summed counts (r1)
    [2] -- Mean ratio based on weighted means of block-by-block data*
        * [2] is omitted when using only 1 block (same as [0])

    Keyword args:
    N -- Number of filaments to average
    blocks -- Number of blocks to run
    cycles -- Number of measurements within a block
    Y_time -- Time spent measuring isotope Y (seconds)
    X_time -- Time spent measuring isotope X (seconds)
    Y -- Count rate of Y (counts/second)
    X -- Count rate of X (counts/second)
    block_time -- Time for each block (seconds)
    beam_decay_pct_min -- Beam decay rate (percent/minute)

"""
    if (blocks==1):  # 1-block analysis is the same as not using block-by-block
        results = find_bias(N,cycles,Y_time,X_time,Y,X,block_time-X_time-Y_time,\
                         beam_decay_pct_min)
        return results[[0,2]]  # Make the results match return formatting

    # Initialize array of results
    results = np.double([0,0,0])
    for i in range(N):
        # Run the simulations, summing the results
        results = results + analyze_blocks(blocks, cycles, Y_time, X_time,Y, X,\
                                           block_time, beam_decay_pct_min)
    results/=N  # Compute the average from the summed totals
    return results

def find_bias(N=10000,cycles=80,Y_time=3,X_time=3,Y=50000,X=50000,\
              other_time=0, beam_decay_pct_min=0):
    """Acquire statistics on a large number of virtual experiments

    Returns a list of values:
    [0] -- Average of mean of measured ratios
    [1] -- Relative discrepancy from the true value
    [2] -- Average of ratio of total counts
    [3] -- Relative discrepancy from the true value

    Keyword args:
    N -- Number of filaments to average
    blocks -- Number of blocks to run
    cycles -- Number of measurements within a block
    Y_time -- Time spent measuring isotope Y (seconds)
    X_time -- Time spent measuring isotope X (seconds)
    Y -- Count rate of Y (counts/second)
    X -- Count rate of X (counts/second)
    block_time -- Time for each block (seconds)
    beam_decay_pct_min -- Beam decay rate (percent/minute)

"""
    mysample=np.double([0,0,0,0])
    for i in range(N):
        mysample = mysample + run_filament(cycles,Y_time,X_time,Y,X)

    mysample/=N
    return mysample

