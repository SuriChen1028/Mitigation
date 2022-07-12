# File Description

This folder stores files used to compute post damage jump results on mercury cluster

- `post_damage.py` is the python file
The flags are as follows:
```
usage: post_damage_one_tech.py [-h] [--xi_a XI_A] [--xi_g XI_G] [--id ID] [--psi_0 PSI_0] [--psi_1 PSI_1]

xi_r values

optional arguments:
  -h, --help     show this help message and exit
  --xi_a XI_A
  --xi_g XI_G
  --id ID
  --psi_0 PSI_0
  --psi_1 PSI_1
```
- `submit.sh` is the sbatch submission file that loop over all the 
$\gamma_3$
values

## Usage

First, make sure `post_damage.py` and `submit.sh` are under the root directory of repository on the cluster.
Or, move the file using the following command:
```
mv post_damage.py ../post_damage.py
mv submit.sh ../submit.sh
```

To submit the batch job, use the file in the following way:
```
chmod +x submit.sh
```
to make it executable. The use flags to set up 
$\psi_0$
and
$\psi_1$
values:
```
./submit.sh -0 <YourChoiceofPsi_0Value> -1 <YourChoiceofPsi_1value>
```

