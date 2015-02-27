# PTopt
Method to optimize temperature spacings of Parallel Tempering simulations.

This algorithm is published in:
A.J. Ballard and D.J. Wales, J. Chem. Theor. Comput., 10, 5599-5605 (2014). 
Superposition-Enhanced Estimation of Optimal Temperature Spacings for Parallel Tempering Simulations 

The method relies upon a database of configurational minima of the underlying energy landscape. Required inputs are 
1) a list of energies
2) a list of entropies
for the configurational minima under consideration.

An example for the LJ31 cluster is included.
