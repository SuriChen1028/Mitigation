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
- [`./for_mercury/`](./for_mercury/) contains code that used on Mercury.
- [`Abatement.ipynb`](./Abatement.ipynb) contains code that load HJB solutions and simulate trajectories of emission, R&D investment and physical investment.
- [`./src/`](./src/) contains source files that solve 2D and 3D HJBs, simulate trajectories.
- [`Plots_compare_psi_0.ipynb`](./Plots_compare_psi_0.ipynb) plots trajectories fixing value of
$\psi_1$
and compare the impact of
$\psi_0$
.
- [`Plots_compare_psi_1.ipynb`](./Plots_compare_psi_1.ipynb) plots trajectories fixing value of 
$\psi_0$ 
and compare the impact of 
$\psi_1$
.
- [`res_data/6damage/`](./res_data/6damage/) contains more recent solutions of HJBs under different values of 
$\psi_0$
and
$\psi_1$
. Under the same `psi_0_x.xxx_psi_1_x.xxx` folder, results of some 
$\xi_a$
and
$\xi_g$
configurations are also available.
- [`./res_data_old_files`](./res_data_old_files/): data from April, 2022 are stored here.
	- [`./res_data_old_files/10damage`](./res_data_old_files/20damage/): parameter setting from April, 2022, with 10 values of 
$\gamma_3$
.
	- [`./res_data_old_files/20damage`](./res_data_old_files/20damage/): parameter setting from April, 2022, with 20 values of
$\gamma_3$
.
- [`./Plots_xi_comparison.ipynb`](./Plots_xi_comparison.ipynb): code for plots from April, 2022.
- [`model144.csv`](./model144.csv) contains the 144 climate sensitivity parameters values.
