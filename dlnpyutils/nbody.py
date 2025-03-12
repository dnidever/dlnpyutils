import numpy as np
import matplotlib.pyplot as plt

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

# Modifications by D. Nidever   March 2025

def getAcc( pos, mass, G, softening ):
    """
    Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations 
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass
    
    # pack together the acceleration components
    a = np.hstack((ax,ay,az))

    return a
	
def getEnergy( pos, vel, mass, G ):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum( mass * vel**2 ))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations 
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))
	
    return KE, PE;


def run(init,tend=10.0,dt=0.01,softening=0.1):
    """ N-body simulation """
	
    # Simulation parameters
    t = 0         # current time of the simulation
    G = 1.0       # Newton's Gravitational Constant
    N = len(init)  # Number of particles
    
    # Generate Initial Conditions
    np.random.seed(17)            # set the random number generator seed

    mass = init['mass'].reshape(-1,1)
    pos = init['pos']
    vel = init['vel']
	
    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel,0) / np.mean(mass)
	
    # calculate initial gravitational accelerations
    acc = getAcc( pos, mass, G, softening )
	
    # calculate initial energy of system
    KE, PE  = getEnergy( pos, vel, mass, G )
	
    # number of timesteps
    Nt = int(np.ceil(tend/dt))
	
    # save energies, particle orbits for plotting trails
    dtype = [('time',float),('mass',float,N),('pos',float,(N,3)),
             ('vel',float,(N,3)),('KE',float),('PE',float)]
    tab = np.zeros(Nt,dtype=np.dtype(dtype))
    tab['time'][0] = 0.0
    tab['mass'][0] = mass[:,0]
    tab['pos'][0] = pos
    tab['vel'][0] = vel
    tab['KE'][0] = KE
    tab['PE'][0] = PE
    
    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2.0
		
        # drift
        pos += vel * dt
		
        # update accelerations
        acc = getAcc( pos, mass, G, softening )
		
        # (1/2) kick
        vel += acc * dt/2.0
		
        # update time
        t += dt
		
        # get energy of system
        KE, PE  = getEnergy( pos, vel, mass, G )

        tab['time'][i] = t
        tab['mass'][i] = mass[:,0]
        tab['pos'][i] = pos
        tab['vel'][i] = vel
        tab['KE'][i] = KE
        tab['PE'][i] = PE
        
    return tab

def example():
    """
    Run Example N-body simulation
    """

    # Simulation parameters
    N         = 100    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = 10.0   # time at which simulation ends
    dt        = 0.01   # timestep
    softening = 0.1    # softening length

    # Generate Initial Conditions
    np.random.seed(17)            # set the random number generator seed
	
    mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
    pos  = np.random.randn(N,3)   # randomly selected positions and velocities
    vel  = np.random.randn(N,3)
    dtype = [('mass',float),('pos',float,3),('vel',float,3)]
    init = np.zeros(N,dtype=np.dtype(dtype))
    init['mass'] = mass[:,0]
    init['pos'] = pos
    init['vel'] = vel
    
    tab = run(init,tend=tEnd,dt=dt,softening=softening)
    
    return tab

def solarsystem():
    """
    Simulated the solar system
    """

    # Sun
    # Mercury
    # Venus
    # Earth
    # Moon
    # Mars
    # Jupiter
    # Saturn
    # Uranus
    # Neptune

    Msun = 1.988e30    # kg
    Mearth = 5.972e24  # kg
    Mjup = 1.898e27    # kg
    AU = 1.496e8       # km
    
    # Simulation parameters
    N         = 10     # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = 10.0   # time at which simulation ends
    dt        = 0.01   # timestep
    softening = 0.001  # softening length

    # Generate Initial Conditions
    np.random.seed(17)            # set the random number generator seed
	
    # Everything orbit counter-clockwise as seen from
    # the north ecliptic pole
    dtype = [('mass',float),('pos',float,3),('vel',float,3)]
    init = np.zeros(N,dtype=np.dtype(dtype))
    # Sun
    init['mass'][0] = Msun
    init['pos'][0,:] = 0
    init['vel'][0,:] = 0
    # Mercury
    init['mass'][1] = 3.285e23              # kg
    init['pos'][1,:] = [0.387*AU, 0.0, 0.0]
    init['vel'][1,:] = [0.0, -47.9, 0.0]    # km/s
    # Venus
    init['mass'][2] = 4.867e24              # kg
    init['pos'][2,:] = [0.72*AU, 0.0, 0.0]
    init['vel'][2,:] = [0.0, -35.02, 0.0]   # km/s
    # Earth
    init['mass'][3] = Mearth
    init['pos'][3,:] = [AU, 0.0, 0.0]       # km
    init['vel'][3,:] = [0.0, -30, 0.0]      # km/s
    # Moon, motion relative to the earth
    init['mass'][4] = 7.347e22              # kg
    init['pos'][4,:] += init['pos'][3,:]
    init['pos'][4,:] = [3.844e5, 0.0, 0.0]  # km
    init['vel'][4,:] += init['vel'][3,:]
    init['vel'][4,:] = [0.0, -1.022, 0.0]   # km/s
    # Mars
    init['mass'][5] = 6.389e23              # kg
    init['pos'][5,:] = [1.52*AU, 0.0, 0.0]  # km
    init['vel'][5,:] = [0.0, -24.077, 0.0]  # km/s
    # Jupiter
    init['mass'][6] = Mjup
    init['pos'][6,:] = [5.2*AU, 0.0, 0.0]   # km
    init['vel'][6,:] = [0.0, -13.1, 0.0]    # km/s
    # Saturn
    init['mass'][7] = 5.683e26              # kg
    init['pos'][7,:] = [9.5*AU, 0.0, 0.0]   # km
    init['vel'][7,:] = [0.0, -9.64, 0.0]    # km/s
    # Uranus
    init['mass'][8] = 8.681e25              # kg
    init['pos'][8,:] = [19.2*AU, 0.0, 0.0]  # km
    init['vel'][8,:] = [0.0, -6.81, 0.0]    # km/s
    # Neptune
    init['mass'][9] = 1.024e26              # kg
    init['pos'][9,:] = [30.1*AU, 0.0, 0.0]  # km
    init['vel'][9,:] = [0.0, -5.43, 0.0]    # km/s
    
    tab = run(init,tend=tEnd,dt=dt,softening=softening)
    
    return tab
    

def main():
    """ N-body simulation """
	
    # Simulation parameters
    N         = 100    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = 10.0   # time at which simulation ends
    dt        = 0.01   # timestep
    softening = 0.1    # softening length
    G         = 1.0    # Newton's Gravitational Constant
    plotRealTime = True # switch on for plotting as the simulation goes along
	
    # Generate Initial Conditions
    np.random.seed(17)            # set the random number generator seed
	
    mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
    pos  = np.random.randn(N,3)   # randomly selected positions and velocities
    vel  = np.random.randn(N,3)
	
    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel,0) / np.mean(mass)
	
    # calculate initial gravitational accelerations
    acc = getAcc( pos, mass, G, softening )
	
    # calculate initial energy of system
    KE, PE  = getEnergy( pos, vel, mass, G )
	
    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))
	
    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N,3,Nt+1))
    pos_save[:,:,0] = pos
    KE_save = np.zeros(Nt+1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt+1)
    PE_save[0] = PE
    t_all = np.arange(Nt+1)*dt
	
    # prep figure
    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])
	
    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2.0
		
        # drift
        pos += vel * dt
		
        # update accelerations
        acc = getAcc( pos, mass, G, softening )
		
        # (1/2) kick
        vel += acc * dt/2.0
		
        # update time
        t += dt
		
        # get energy of system
        KE, PE  = getEnergy( pos, vel, mass, G )
		
        # save energies, positions for plotting trail
        pos_save[:,:,i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE
		
        # plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:,0,max(i-50,0):i+1]
            yy = pos_save[:,1,max(i-50,0):i+1]
            plt.scatter(xx,yy,s=1,color=[.7,.7,1])
            plt.scatter(pos[:,0],pos[:,1],s=10,color='blue')
            ax1.set(xlim=(-2, 2), ylim=(-2, 2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-2,-1,0,1,2])
            ax1.set_yticks([-2,-1,0,1,2])
			
            plt.sca(ax2)
            plt.cla()
            plt.scatter(t_all,KE_save,color='red',s=1,label='KE' if i == Nt-1 else "")
            plt.scatter(t_all,PE_save,color='blue',s=1,label='PE' if i == Nt-1 else "")
            plt.scatter(t_all,KE_save+PE_save,color='black',s=1,label='Etot' if i == Nt-1 else "")
            ax2.set(xlim=(0, tEnd), ylim=(-300, 300))
            ax2.set_aspect(0.007)
			
            plt.pause(0.001)

    # add labels/legend
    plt.sca(ax2)
    plt.xlabel('time')
    plt.ylabel('energy')
    ax2.legend(loc='upper right')
	
    # Save figure
    plt.savefig('nbody.png',dpi=240)
    plt.show()
	    
    return 0

  
if __name__== "__main__":
    main()
