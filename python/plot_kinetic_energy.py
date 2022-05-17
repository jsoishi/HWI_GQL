"""plot_kinetic_energy.py

python3 plot_kinetic_energy.py <run_dir>

where <run_dir> is the run directory.

"""
import h5py
import matplotlib.pyplot as plt
import sys

run_dir = sys.argv[-1]
outdir = run_dir.lstrip("runs/run_").strip("/")

with h5py.File(run_dir+"timeseries/timeseries_s1/timeseries_s1_p0.h5","r") as ts:
    t = ts['scales/sim_time'][:]
    KE = ts['tasks/KE'][:,0,0]
    KE_p = ts['tasks/KE_pert'][:,0,0]

plt.subplot(111)
plt.plot(t,KE_p)
plt.xlabel("t")
plt.ylabel("perturbation KE")
plt.savefig("../figs/"+outdir+"_KE_pert.png",dpi=300)
