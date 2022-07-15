# Mitigation
Repository for mitigation model with endogenous R&amp;D investments.

- [`post_damage_one_tech.py`](./post_damage_one_tech.py) computes post damage HJBs with a given value of 
$\gamma_3$
both post tech and pre tech, and a given set of uncertainty parameters 
$\xi_a$
and
$\xi_g$
- [`pre_damage.py`](./pre_damage.py) computes pre damage HJBs with a given configuration of uncertainty parameters
$\xi_a$
,
$\xi_g$
and
$\xi_p$.

- [`./notes/`](./notes/) includes notes about the model we are solving;
- [`./figures`](./figures/) includes our previous plots with 20 damages and 2 technology jump, and also recents experiments on different combination of 
$\psi_0$
and
$\psi_1$
.
- [`./for_mercury/`](./mercury/) contains code that used on Mercury.
- `Abatement.ipynb` contains code that load HJB solutions and simulate trajectories of emission, R&D investment and physical investment.
- [`./src/`] contains source files that solve 2D and 3D HJBs, simulate trajectories.
