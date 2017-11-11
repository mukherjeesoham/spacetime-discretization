### Adding Physics
---
    + Formulate and implement the problem as the variation of a discretized action for the scalar wave equation [D
    + Implement null coordinates [D]
    + Implement a multi-patch system for computing the entire domain piecewise [D]
    + Implement futures for asynchronous computing of the patches [D]
    + Consistently include the potential in the code [D]
    + Implement a single patch to a global patch structure for computing with arbitrary potentials [W]
    + Solve in the conformally compactified Minkowski spacetime [R]
    + Formulate the problem in terms of action with Lagrange multipliers for boundary conditions
    + Find a matrix algebra formulation for GR

### Consistency Checks
---
    + Check for h-p convergence [W]
    + Check for convergence of the action for the solution [W]
    + Check for Eigenvalues (max/min) [D]
    + Check for the higher mode coefficients [W]
    + Compute energy flux at the boundary and energy on the patch to check for energy conservation.
    + Implement and apply a filter for non-linear problems.

### Code optimization 
---
    + Remove large matrix storage
    + Do modal to nodal transformations direction-wise.
    + Implement classes in the Python code
    + Port the code in Haskell
    + Port the code in Julia
    + Do benchmarking and scaling tests
    + Use Category theory to handle futures in languages like Haskell.

### Open Issues
---
    + Topology of Scri+ ?
    + Does there exist a conformal dual for spacetime? Concretely, the conformal dual for space (r) 
    is a mapping (1/r). What about spacetime?
    + Use sparse grids in our project? <see https://arxiv.org/abs/1710.09356>
    + To extend beyond 1+1, we need to give away the diamond shaped patches we are using. We need to 
    use higher order simplices to tessellate the domain.
    ...+ Require basis functions on simplices?
    ...+ Use stretched cubes with null faces?
    ...+ Require the language of geometric algebra.
    + Compactification of infinities: Can we use conformal compactification, or use a conformally 'dual' mapping
    to handle infinity on the grid?
    ...+ Use of Bessel functions as basis functions when treating infinity?
    + Imposing boundary conditions at infinity: what is the topology and how do we handle it?
    + What data structure do we use for storing the points on complex simplices? 
    + We need to store the orientation and the values at the points. Can Pachner moves help?
    + Solve for a scalar wave test particle on a background? No conservation of charge like the EM case?
