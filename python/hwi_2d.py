"""
Holmboe Wave Instability script


"""
from configparser import ConfigParser
import sys
from pathlib import Path
import numpy as np
import dedalus.public as d3
import logging

from GQLProjection import GQLProjection as Project
logger = logging.getLogger(__name__)

# Parses .cfg filename passed to script
config_file = Path(sys.argv[-1])

logger.info("Running with config file {}".format(str(config_file)))
# Parse .cfg file to set global parameters for script
runconfig = ConfigParser()
runconfig.read(str(config_file))

datadir = Path("runs") / config_file.stem

params = runconfig['params']
# Parameters
GQL = params.getboolean('GQL')
Lambda_x = params.getint('Lambda_x')
Lx = params.getfloat('Lx')
Lz = params.getfloat('Lz')
Nx = params.getint('Nx')
Nz = params.getint('Nz')
Re = params.getfloat('Re')
Pr = params.getfloat('Pr')
Aspect = params.getfloat('Aspect')
Rib = params.getfloat('Rib')
rho_ampl = params.getfloat('rho_ampl')

run_params = runconfig['run']
restart = run_params.get('restart_file')
stop_wall_time = run_params.getfloat('stop_wall_time')
stop_sim_time = run_params.getfloat('stop_sim_time')
stop_iteration = run_params.getint('stop_iteration')
max_timestep = run_params.getfloat('max_timestep')

dealias = 3/2
timestepper = d3.RK222
dtype = np.float64


# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)
x = dist.local_grid(xbasis)
z = dist.local_grid(zbasis)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
rho = dist.Field(name='rho', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
U = dist.VectorField(coords, name='U', bases=(xbasis,zbasis)) # background velocity
tau_rho1 = dist.Field(name='tau_rho1', bases=xbasis)
tau_rho2 = dist.Field(name='tau_rho2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
tau_p = dist.Field(name='tau_p')

# unit vectors
ex = dist.VectorField(coords, name='ex')
ez = dist.VectorField(coords, name='ez')
ex['g'][0] = 1
ez['g'][1] = 1

integ = lambda A: d3.Integrate(d3.Integrate(A, 'x'), 'z')

lift_basis = zbasis.derivative_basis(1) # First derivative basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # First-order reduction
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_rho = d3.grad(rho) + ez*lift(tau_rho1)

Project_high = lambda A: Project(A,[Lambda_x,],'h')
Project_low = lambda A: Project(A,[Lambda_x,],'l')
ul = Project_low(u)
uh = Project_high(u)
rhol = Project_low(rho)
rhoh = Project_high(rho)

# Problem
problem = d3.IVP([u, rho, p, tau_rho1, tau_rho2, tau_u1, tau_u2, tau_p], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
if GQL:
    problem.add_equation("dt(u) + grad(p) - div(grad_u)/Re + Rib*rho*ez + lift(tau_u2) = -Project_low(ul@grad(ul) + uh@grad(uh)) - Project_high(ul@grad(uh) + uh@grad(ul))")
    problem.add_equation("dt(rho) - div(grad_rho)/(Re*Pr) + lift(tau_rho2) = -Project_low(ul@grad(rhol)+ uh@grad(rhoh)) - Project_high(ul@grad(rhoh) + uh@grad(rhol))")
else:
    problem.add_equation("dt(u) + grad(p) - div(grad_u)/Re + Rib*rho*ez + lift(tau_u2) = -u@grad(u)")
    problem.add_equation("dt(rho) - div(grad_rho)/(Re*Pr) + lift(tau_rho2) = -u@grad(rho)")
problem.add_equation("integ(p) = 0") # Pressure gauge

# boundary conditions
problem.add_equation("rho(z=-Lz/2) = 2")
problem.add_equation("dot(ez, u(z=-Lz/2)) = 0")
problem.add_equation("dot(ex, dot(ez,grad_u(z=-Lz/2))) = 0")

problem.add_equation("rho(z=Lz/2) = 0")
problem.add_equation("dot(ez, u(z=Lz/2)) = 0")
problem.add_equation("dot(ex, dot(ez,grad_u(z=Lz/2))) = 0")

# Solver
solver = problem.build_solver(timestepper)

# Initial conditions
# Background shear
U['g'][0] = np.tanh(z)
u['g'] = U['g']
rho['g'] = (1-np.tanh(Aspect*z)) 

if restart:
    solver.load_state(restart)
else:
    # Add small velocity perturbations localized to the shear layers
    # use vector potential to ensure div(u) = 0
    A = dist.Field(name='A', bases=(xbasis,zbasis))
    A.fill_random('g', seed=42, distribution='normal')
    A.low_pass_filter(scales=(0.2, 0.2, 0.2))
    A['g'] *= (1 - (2*z/Lz)**2) *np.exp(-(z)**2) # Damp noise at walls
    # uncomment next line and comment out following three
    # to perturb rho instead of velocity field
    #rho['g'] += 1e-3*A['g']
    up = d3.skew(d3.grad(A)).evaluate()
    up.change_scales(1)
    u['g'] += 1e-3*up['g']
# setup energies
KE = 0.5 * d3.DotProduct(u,u)
KE.name = 'KE'
u_p = u - U
KE_p = 0.5* d3.DotProduct(u_p,u_p)
KE_p.name = 'KE_p'

solver.stop_sim_time = solver.sim_time + stop_sim_time
if stop_iteration:
    logger.info("Setting stop_iteration to {}".format(stop_iteration))
    solver.stop_iteration = solver.iteration + stop_iteration
# Analysis
if dist.comm.rank == 0:
    if not datadir.exists():
        datadir.mkdir(parents=True)
# checkpoints
check = solver.evaluator.add_file_handler(datadir / Path('checkpoints'), iter=100, max_writes=1, virtual_file=True)
check.add_tasks(solver.state)
# use this later for slices in 3D
snapshots = solver.evaluator.add_file_handler(datadir/'snapshots', sim_dt=1., max_writes=10)
snapshots.add_task(rho)
snapshots.add_task(u)
# timeseries 
timeseries = solver.evaluator.add_file_handler(datadir / Path('timeseries'), sim_dt=0.1)
timeseries.add_task(integ(KE), name='KE')
timeseries.add_task(integ(KE_p), name='KE_pert')
timeseries.add_task(integ(d3.div(u)*d3.div(u))**0.5, name='rms_div_u')
# CFL
CFL = d3.CFL(solver, initial_dt=1e-3, cadence=10, safety=0.2, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.dot(u,ez)**2, name='w2')
flow.add_property(d3.div(u), name='divu')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_w = np.sqrt(flow.max('w2'))
            max_divu = flow.max('divu')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f, max(div(u)=%e' %(solver.iteration, solver.sim_time, timestep, max_w, max_divu))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
    solver.evaluate_handlers_now(timestep)
