# This directory contains source modules used for computation:

- `simulation.py` contains:
	- `simulate_pre` computes trajectories of pre technology jump states and controls
- `solver.py` contains:
	- `pde_one_iteration`: get ready coefficient matrix for `PETSc` solver
	- `_FOC_update`: update controls based on last iteration inputs
	- `hjb_pre_tech`: solve for pre technology jump HJB
		> If argument `V_post_damage=None`, this is solving the post damage jump HJB with given input of
$\gamma_3$
		> If argument `V_post_damag=` is a ndarray of post damage HJB solutions, this is solving pre damage jump solution.
- `supportfunctions.py` contains:
	- `finiteDiff`: computes derivatives up to 4D matrix.
	- `finiteDiff_3D`: computes derivatives up to 3D matrix.
	- `PDESolver`: computes 3D PDE
	- `PDESolver_4D`: computes 4D PDE
