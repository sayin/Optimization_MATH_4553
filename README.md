> # Python code for Optimal Control for 1D Convection-Diffusion Equation

# Abstract
The CD equation describes the ow of heat, particles, or other physical quantities in
situations where there is both diffusion and convection. In this report, we use optimal
control input (u) to steer the temperature of a thin rod undergoing convection-diffusion
(CD) process along with external forcing (f). Hence the cost function to be minimized
consists of con icting objectives that is temperature distribution and amount of
control input (u). This formulation is equivalent to linear quadratic regulator (LQR)
problem. The objective function is reformulated as quadratic programming problem
which has unique analytical solution. Furthermore, we also use gradient descent algo-
rithm to arrive at approximated solution which is compared to analytical solution.


**Some Resutls**

Optimum temperature distribution. |  Optimum control input u(x).
----- | -----
<img src="b11.pdf" width="90%">| <img src="b13.pdf" width="90%" >
