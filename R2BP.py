# ----------------------- Restricted 2-Body Problem Solver -----------------------
#
# ---------------------------------- Description ---------------------------------
# 
#   Restricted 2-body problem numerical solver using a 4th-order Yoshida method
#       > Sympletic integrator preserves Hamiltonian energy close to machine precision
#       > Overkill for the simplicicty of the R2BP (known analytical solution)
#
#   Perturbation model includes only the J2 and third body perturbations
#       > Conservative perturbations preserve the Hamiltonian energy structure
#       > Angular momentum L is not conserved under perturbation acceleration model 
#   Currently only accounts for periodic orbits (0 <= e < 1)
#
# ---------------------------------- References ----------------------------------
# [1] Yoshida, H. (1990). "Construction of higher order symplectic integrators." Physics Letters A, 150(5–7), 262–268.
# [2] Curtis, H.D. (2021). Orbital Mechanics for Engineering Students. 4th ed. Butterworth-Heinemann
# [3] Bate, R.R., Mueller, D.D. & White, J.E. (1971). Fundamentals of Astrodynamics. Dover Publications
# [4] https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg
# [5] https://ssd.jpl.nasa.gov/astro_par.html
# 
# ---------------------------------- Notes ---------------------------------------
#
#   Simulation specifications (observed)
#       > Circular LEO orbits -> stepsPerOrbit ~ 1E3 (|dE| ~ E-12)
#       > Molniya orbits      -> stepsPerOrbit ~ 1E5 (|dE| ~ E-10)
#       > Tundra orbits       -> stepsPerOrbit ~ 1E4 (|dE| ~ E-12)
#
# --------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


print(f""), print(f""), print(f"------------------------- Running -------------------------"), print(f""), print(f"")
s_runtime = time.perf_counter()

# ---------------------------------- Simplifications -----------------------------
cos, sin = np.cos, np.sin


# ---------------------------------- Functions -----------------------------------

# 4th-order Yoshida coefficients
beta = 2.0**(1/3)
c = [1/(2-beta)/2, (1-beta)/(2-beta)/2, (1-beta)/(2-beta)/2, 1/(2-beta)/2]
d = [1/(2-beta), -beta/(2-beta), 1/(2-beta), 0]

def gravity(r, mu):
    # Acceleration due to gravity (2BP)
    r_mag = np.linalg.norm(r)
    return mu/r_mag**3*r

def gravityPerturbed(r, mu): 
    # Acceleration due to gravity with J2 perturbation
    J2 = 1.08263E-3 
    r_mag = np.linalg.norm(r)

    a_J2 = 1.5*J2*mu*r_Earth**2/r_mag**5 * np.array([r[0]*(5*(r[2]/r_mag)**2 - 1), r[1]*(5*(r[2]/r_mag)**2 - 1), r[2]*(5*(r[2]/r_mag)**2 - 3)])
    
    return gravity(r, mu) + a_J2

def yoshida4(r, v, dt, func_accel):
    # 4th-order Yoshida step
    for i in range(4):
        r = r + c[i]*dt*v
        a = func_accel(r)
        v = v - d[i]*dt*a
    return r, v

def integrate(r0, v0, dt, n, mu, perturbed):
    # Numerical integrator (using 4th-order Yoshida)
    if perturbed:
        # Perturbed 2-body problem with J2 effect
        func_accel = lambda r: gravityPerturbed(r, mu)  
    else:
        # Unperturbed 2-body problem
        func_accel = lambda r: gravity(r, mu)   
              
    def energy(r,v, r_Earth, perturbed):
        U_2BP = -mu/np.linalg.norm(r)

        if perturbed:
            J2 = 1.08263E-3
            U_2BP_perturbed = -0.5*J2*mu*r_Earth**2/np.linalg.norm(r)**3 * (3*(r[2]/np.linalg.norm(r))**2 - 1)
            U = U_2BP + U_2BP_perturbed
        else:
            U = U_2BP

        return 0.5*np.linalg.norm(v)**2 + U
    
    def angMom(r,v):
        return np.linalg.norm(np.cross(r, v))
    
    # Initial conditions
    r , v = r0.copy(), v0.copy()    
    trajectory = np.zeros((n+1, 8))
    trajectory[0] = np.concatenate([r,v, [energy(r,v, r_Earth, perturbed)], [angMom(r,v)]])

    # Yoshida loop for "trajectory" state vector
    for i in range(n):
        r, v = yoshida4(r, v, dt, func_accel)
        trajectory[i+1] = np.concatenate([r,v, [energy(r,v, r_Earth, perturbed)], [angMom(r,v)]])
        
    # Results (using Pandas data frame)
    results = pd.DataFrame(trajectory, columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'E', 'L'])
    results['t'] = np.arange(len(results))*dt

    return results

def OEtoRV(a, e, i, raan, argp, theta, mu):
    # Orbital elements to r(t) and v(t)
    # Perifocal reference frame results
    h = np.sqrt(a*mu*(1 - e**2))
    r = h**2/mu/(1 + e*cos(theta)) * np.array([cos(theta), sin(theta), 0])
    v = mu/h * np.array([-sin(theta), e + cos(theta), 0])

    # Rotation matrix from perifocal to inertial frame (ECI)
    R_PI = np.array([[-sin(raan)*cos(i)*sin(argp)+cos(raan)*cos(argp), cos(raan)*cos(i)*sin(argp)+sin(raan)*cos(argp), sin(i)*sin(argp)], 
                     [-sin(raan)*cos(i)*cos(argp)-cos(raan)*sin(argp), cos(raan)*cos(i)*cos(argp)-sin(raan)*sin(argp), sin(i)*cos(argp)], 
                     [ sin(raan)*sin(i),                              -cos(raan)*sin(i),                               cos(i)          ]])
    R_IP = np.matrix_transpose(R_PI)

    # Tensor transformation law (rank-1)
    return R_IP @ r, R_IP @ v

def UTCtoJ0(date):
    # Date & Time in UTC to Julian day number at 0h UTC
    # Valid for any year between 1900 to 2100
    return 367*date[2] - int(7/4*(date[2]+int((date[0] + 9)/12))) + int(275*date[0]/9) + date[1] + 1721013.5

def JD(timeUTC, J0):
    # Time (UTC) to Julian day
    return J0 + (timeUTC[0] + timeUTC[1]/60 + timeUTC[2]/3600)/24

def GST(J0, timeUTC):
    # Grennwich sidereal time ()
    T0 = (J0- 2451545)/36525
    theta_G0 = (100.4606184 + 36000.77004*T0 + 0.000387933*T0**2 - 2.583E-8 *T0**3) % 360
    theta_G = (theta_G0 + 360.985864724 * (timeUTC[0] + timeUTC[1]/60 + timeUTC[2]/3600)/24) % 360
    return np.deg2rad(theta_G) 

def latLongECEF(traj, theta_G):
    # Coordinate transformation from Earth-Centered Inertial RF to Earth-Centered Earth-Fixed RF
    t = traj['t'].values
    r_ECI = np.array([traj['x'].values, traj['y'].values, traj['z'].values])

    Omega_Earth = 7.292115900231276e-05     # Rotational rate of Earth [rad/s]
    theta_G = theta_G + Omega_Earth*t       # Greenwich Sidereal Time [rad]

    # Multplied out from 1-tensor transformation law
    x_ECEF = r_ECI[0]*cos(theta_G) + r_ECI[1]*sin(theta_G) + r_ECI[2]*0
    y_ECEF = r_ECI[0]*-sin(theta_G) + r_ECI[1]*cos(theta_G) + r_ECI[2]*0
    z_ECEF = r_ECI[2]

    # From Cartesian = spherical coords -> Solving for latitude and longitude angles
    r_mag = np.sqrt(x_ECEF**2 + y_ECEF**2 + z_ECEF**2)
    lat = np.degrees(np.arcsin(z_ECEF/r_mag))
    long = np.degrees(np.arctan2(y_ECEF, x_ECEF))

    return lat, long


# ---------------------------------- Inputs --------------------------------------

# Orbtial elements
zp = 32623.6                    # Periapsis altitude [km]               
e = 0.075                       # Eccentricity []
i = np.radians(43.0)            # Inclination [rad] (input in degrees)
raan = np.radians(195.0)        # Right ascension of ascending node [rad] (input in degrees)
argp = np.radians(270.0)        # Argument of periapsis [rad] (input in degrees)
theta = np.radians(305.0)       # True anomaly [rad] (input in degrees)

# Initial epoch (UTC)
date = np.array([12, 26, 2009])  # Date [mm/dd/yyyy]
timeUTC = np.array([12, 0, 0])  # Universal Time Coordinated (UTC) [hh:mm:ss]

# Earth parameters 
mu = 398600.435507              # Gravitational parameter [km^3/s^2]
r_Earth = 6378.1366             # Radius of Earth [km]

# Simulation specifications
stepsPerOrbit = 10000           # Steps per orbit []
numOrbits = 1.5                 # Number of orbits []
perturbed = True                # Whether to include perturbation model (True/False)


# ---------------------------------- Calculations ---------------------------------

# Simulation setup 
a = (zp+r_Earth)/(1-e)
T = 2*np.pi/np.sqrt(mu)*a**1.5
dt = T/stepsPerOrbit
n = int(stepsPerOrbit*numOrbits)

# Initialization
r0, v0 = OEtoRV(a, e, i, raan, argp, theta, mu)

# Full 4th-order Yoshida solve
traj = integrate(r0, v0, dt, n, mu, perturbed)

# Sidereal time parameters
J0 = UTCtoJ0(date)
julian_date = JD(timeUTC, J0)
theta_G = GST(J0, timeUTC)

# Latitude and longitude (initial and final)
lat, long = latLongECEF(traj, theta_G)
startingLatitude = np.degrees(np.arcsin(r0[2]/np.linalg.norm(r0)))
startingLongitude = np.degrees(np.arctan2(r0[1], r0[0])) - np.rad2deg(theta_G) % 360
finalLatitude = lat[-1]
finalLongitude = long[-1]
if startingLongitude < 0: startingLongitude += 360
elif startingLongitude > 180: startingLongitude -= 360
else: pass
if finalLongitude < 0: finalLongitude += 360
elif finalLongitude > 180: finalLongitude -= 360
else: pass

# Max errors (energy, angular momentum)
maxDeltaE = (traj['E'] - traj['E'].iloc[0]).abs().max()
maxDeltaL = (traj['L'] - traj['L'].iloc[0]).abs().max()

# Simulation runtime
e_runtime = time.perf_counter()
runtime = e_runtime - s_runtime

# ---------------------------------- Plotting -------------------------------------

def orbit3D(traj):
    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # Earth sphere — low resolution to reduce lag
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = r_Earth * np.outer(np.cos(u), np.sin(v))
    y_sphere = r_Earth * np.outer(np.sin(u), np.sin(v))
    z_sphere = r_Earth * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere,
                      linewidth=0.2, color='cornflowerblue',
                      alpha=1, rstride=2, cstride=2)

    # Orbit trajectory — downsampled to reduce lag
    step = max(1, len(traj) // 5000)
    ax.plot(traj['x'].values[::step],
            traj['y'].values[::step],
            traj['z'].values[::step],
            linewidth=0.8, color='royalblue', label='Orbit')

    # Start and end markers
    ax.scatter(*traj[['x','y','z']].iloc[0].values,
               color='green', s=50, zorder=5, label='Start')
    ax.scatter(*traj[['x','y','z']].iloc[-1].values,
               color='red',   s=50, zorder=5, label='End')

    # Equal axis scaling
    max_range = np.array([traj['x'].max() - traj['x'].min(),
                          traj['y'].max() - traj['y'].min(),
                          traj['z'].max() - traj['z'].min()]).max() / 2

    mid_x = (traj['x'].max() + traj['x'].min()) / 2
    mid_y = (traj['y'].max() + traj['y'].min()) / 2
    mid_z = (traj['z'].max() + traj['z'].min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title('Orbital Trajectory')
    ax.legend()

def groundTrack(traj, theta_G):
    # Ground track of satellite (Earth-Centered Earth-Fixed RF) 
    lat, long = latLongECEF(traj, theta_G)

    # Wrapping from 180deg -> -180deg
    long_plot = long.copy()
    wrap = np.abs(np.diff(long_plot)) > 180
    long_plot[1:][wrap] = np.nan

    fig, ax = plt.subplots(figsize=(14,7))

    # Mercator projection background (Reference [4])
    try: 
        img = plt.imread('earth.jpg')
        ax.imshow(img, extent=[-180, 180, -90, 90], aspect='auto', zorder=0)
    except:
        print(f"earth.jpg not found. Proceeding with plotting without background.")

    ax.plot(long_plot, lat, linewidth=1.0, c='yellow')

    ax.scatter(long_plot[0], lat[0], color='green', s=50, zorder=5, label='Start')
    ax.scatter(long_plot[-1], lat[-1], color='red', s=50, zorder=5, label='End')

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title('Ground Track of Satellite')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True)
    ax.legend()

def plotErrors(traj):
    # Hamiltoninan energy & angular momentum errors from initial
    dE = (traj['E'] - traj['E'].iloc[0])
    dL = (traj['L'] - traj['L'].iloc[0])
    t = traj['t'].values

    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(t, dE, linewidth=0.5, c='royalblue')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Hamiltonian Energy Error (ΔH) [km^2/s^2]')
    ax.set_title('Hamiltonian Energy Error (ΔH) vs Time')
    ax.grid(True)

    fig2, ax2 = plt.subplots(figsize=(14,7))
    ax2.plot(t, dL, linewidth=0.5, c='royalblue')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular Momentum Error (ΔL) [km^2/s^2]')
    ax2.set_title('Angular Momentum Error (ΔL) vs Time')
    ax2.grid(True)


# ---------------------------------- Output --------------------------------------- 

# Calculation results
print(f"Simulation time:                     {runtime:.3f} seconds"), print(f"")
print(f"Perturbed:                           {perturbed}")
print(f"Total orbits:                        {numOrbits}")
print(f"Semi-major axis (a):                 {a:.3f} km")
print(f"Orbital period (T):                  {T:.3f} seconds")
print(f"Time step (dt):                      {dt:.3f} seconds")

print(f""), print(f""), print(f"------------------------ Results --------------------------"), print(f""), print(f"")

print(f"Initial epoch:                       {date[0]:02d}/{date[1]:02d}/{date[2]:02d} {timeUTC[0]:02d}:{timeUTC[1]:02d}:{timeUTC[2]:02d} UTC")
print(f"Julian date at epoch:                {julian_date}")
print(f"Greenwich sidereal time (epoch):     {np.rad2deg(theta_G):.3f} deg"), print(f"")

print(f"Starting latitude:                   {startingLatitude:.3f} deg")
print(f"Starting longitude:                  {startingLongitude:.3f} deg")
print(f"Final latitude:                      {finalLatitude:.3f} deg")
print(f"Final longitude:                     {finalLongitude:.3f} deg"), print(f"")

print(f"Max |ΔH|:                            {maxDeltaE:.3e} km^2/s^2")
print(f"Mean |ΔH|:                           {(traj['E'] - traj['E'].iloc[0]).abs().mean():.3e} km^2/s^2")
print(f"Max |ΔL|:                            {maxDeltaL:.3e} km^2/s")
print(f"Mean |ΔL|:                           {(traj['L'] - traj['L'].iloc[0]).abs().mean():.3e} km^2/s^2")

# Plotting
print(f""), print(f""), print(f"------------------------ See Plots ------------------------"), print(f""), print(f"")
orbit3D(traj)
groundTrack(traj, theta_G)
plotErrors(traj)
plt.show()

print(f""), print(f""), print(f"--------------------- Script Concluded --------------------"), print(f""), print(f"")


# ---------------------------------- Examples -------------------------------------

# Example LEO SSO  -------------------------------------

# Orbtial elements
# zp = 520                         # Periapsis altitude [km]               
# e = 0.0                          # Eccentricity []
# i = np.radians(97.45)            # Inclination [rad] (input in degrees)
# raan = np.radians(0.0)           # Right ascension of ascending node [rad] (input in degrees)
# argp = np.radians(0.0)           # Argument of periapsis [rad] (input in degrees)
# theta = np.radians(0.0)          # True anomaly [rad] (input in degrees)

# Initial epoch (UTC)
# date = np.array([12, 1, 2022])   # Date [mm/dd/yyyy]
# timeUTC = np.array([12, 0, 0])   # Universal Time Coordinated (UTC) [hh:mm:ss]
# Earth parameters 
# mu = 398600.435507               # Gravitational parameter [km^3/s^2]
# r_Earth = 6378.1366              # Radius of Earth [km]

# Simulation specifications
# stepsPerOrbit = 1000             # Steps per orbit []
# numOrbits = 5                    # Number of orbits []
# perturbed = True                 # Whether to include J2 perturbation (True/False)


# Example Molniya --------------------------------------

# Orbtial elements
# zp = 500                         # Periapsis altitude [km]               
# e = 0.74                         # Eccentricity []
# i = np.radians(63.43)            # Inclination [rad] (input in degrees)
# raan = np.radians(0.0)           # Right ascension of ascending node [rad] (input in degrees)
# argp = np.radians(270.0)         # Argument of periapsis [rad] (input in degrees)
# theta = np.radians(0.0)          # True anomaly [rad] (input in degrees)

# Initial epoch (UTC)
# date = np.array([12, 1, 2022])   # Date [mm/dd/yyyy]
# timeUTC = np.array([12, 0, 0])   # Universal Time Coordinated (UTC) [hh:mm:ss]

# Earth parameters 
# mu = 398600.435507               # Gravitational parameter [km^3/s^2]
# r_Earth = 6378.1366              # Radius of Earth [km]

# Simulation specifications
# stepsPerOrbit = 10000            # Steps per orbit []
# numOrbits = 5                    # Number of orbits []
# perturbed = True                 # Whether to include J2 perturbation (True/False)


# Quasi-Zenith Satellite System [5] (Tundra) -----------
# https://en.wikipedia.org/wiki/Quasi-Zenith_Satellite_System

# Orbtial elements
# zp = 32623.6                     # Periapsis altitude [km]               
# e = 0.075                        # Eccentricity []
# i = np.radians(43.0)             # Inclination [rad] (input in degrees)
# raan = np.radians(195.0)         # Right ascension of ascending node [rad] (input in degrees)
# argp = np.radians(270.0)         # Argument of periapsis [rad] (input in degrees)
# theta = np.radians(305.0)        # True anomaly [rad] (input in degrees)

# Initial epoch (UTC)
# date = np.array([12, 26, 2009])  # Date [mm/dd/yyyy]
# timeUTC = np.array([12, 0, 0])   # Universal Time Coordinated (UTC) [hh:mm:ss]

# Earth parameters 
# mu = 398600.435507               # Gravitational parameter [km^3/s^2]
# r_Earth = 6378.1366              # Radius of Earth [km]

# Simulation specifications
# stepsPerOrbit = 5000             # Steps per orbit []
# numOrbits = 10                   # Number of orbits []
# perturbed = True                 # Whether to include J2 perturbation (True/False)