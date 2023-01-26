import numpy as np
from dolfinx.fem import *
from dolfinx.mesh import create_interval, exterior_facet_indices, locate_entities, meshtags
from ufl import (TrialFunction, TestFunction, dx, ds, dS, dot, inner, jump, div, grad,
                 lhs, rhs, as_vector, FacetNormal, Measure)
from ufl.algebra import Abs
from ufl.operators import sqrt

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

def steady_adv_diff_1d(mu, order, nx, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    # mesh of 10 uniform linear elements
    mesh_1d = create_interval(MPI.COMM_WORLD, nx, (0.0, 1.0))

    h = 1.0/nx
    adv = 1.0

    Pe = adv*h/(2*mu)
    print('Peclet number = ', Pe)
    
#     P1 = FiniteElement('Lagrange', mesh_1d.ufl_cell(), 1)
    V = FunctionSpace(mesh_1d, ('CG', 1))
    u = TrialFunction(V)
    w = TestFunction(V)
    
    dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
        
    uL, uR = Function(V), Function(V)
    uL.vector.array = left_bc
    uR.vector.array = right_bc

    bcL, bcR = dirichletbc(uL, dofs_L), dirichletbc(uR, dofs_R)

    F = adv*w*u.dx(0)*dx + mu*w.dx(0)*u.dx(0)*dx - w*s*dx

    a, L = lhs(F), rhs(F)

    problem = petsc.LinearProblem(a, L, bcs=[bcL, bcR])
    u = problem.solve()
    
    return u, V.tabulate_dof_coordinates().T[0] #vertex_to_dof_map(V)

def solution_fig2_1(x_space, mu):
    adv = 1
    
    return 1.0/adv*( x_space - (1.0-np.exp(adv/mu*x_space))/(1.0 - np.exp(adv/mu)) )

def steady_adv_diff_1d_SU(mu, order, nx, mesh_1d, method, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    
    h = 1.0/nx
    adv = 1.0
    
    Pe = adv*h/(2*mu)
    
    if method=='full_upwind':
        beta = 1.0
    elif method=='optimal':
        beta = 1.0/np.tanh(Pe) - 1.0/Pe 
    
    print('Peclet number = ', Pe)

    V = FunctionSpace(mesh_1d, ('CG', order))
    u = TrialFunction(V)
    w = TestFunction(V)

    dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
        
    uL, uR = Function(V), Function(V)
    uL.vector.array = left_bc
    uR.vector.array = right_bc
    
    bcL, bcR = dirichletbc(uL, dofs_L), dirichletbc(uR, dofs_R)
    
    F = (w + beta*h/2*w.dx(0) )*adv*u.dx(0)*dx + mu*w.dx(0)*u.dx(0)*dx - w*s*dx

    a, L = lhs(F), rhs(F)

    problem = petsc.LinearProblem(a, L, bcs=[bcL, bcR])
    u = problem.solve()
    
    return u, V.tabulate_dof_coordinates().T[0]

def steady_adv_diff_1d_SUPG(mu, order, nx, mesh_1d, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
        
    h = 1.0/nx/order
    adv = 1.0
    
    Pe = adv*h/(2*mu)
    beta = 1.0/np.tanh(Pe) - 1.0/Pe 
    
    print('Peclet number = ', Pe)

    V = FunctionSpace(mesh_1d, ('CG', order))
    u = TrialFunction(V)
    w = TestFunction(V)

    dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
        
    uL, uR = Function(V), Function(V)
    uL.vector.array = left_bc
    uR.vector.array = right_bc
    
    bcL, bcR = dirichletbc(uL, dofs_L), dirichletbc(uR, dofs_R)
    
    v_norm = dot(adv, adv)
    
    tao = 0.5*adv*beta*h/v_norm # p. 61
    
    adv = Constant(mesh_1d, ScalarType(adv))

    F = w*adv*u.dx(0)*dx + mu*w.dx(0)*u.dx(0)*dx - w*s*dx \
        + adv*tao*w.dx(0)*(adv*u.dx(0) - div(mu*grad(u)) - s)*dx
    # Eq. (2.55) (2.56) （2.57）

    a, L = lhs(F), rhs(F)

    problem = petsc.LinearProblem(a, L, bcs=[bcL, bcR])
    u = problem.solve()
    
    return u, V.tabulate_dof_coordinates().T[0]

def steady_adv_diff_1d_GLS(mu, sigma, order, nx, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    # mesh of 10 uniform linear elements
    mesh_1d = create_interval(MPI.COMM_WORLD, nx, (0.0, 1.0))
    
    h = 1.0/nx
    adv = 1.0
    
    Pe = adv*h/(2*mu)
    beta = 1.0/np.tanh(Pe) - 1.0/Pe 
    
    print('Peclet number = ', Pe)

    V = FunctionSpace(mesh_1d, ('CG', order))
    u = TrialFunction(V)
    w = TestFunction(V)

    dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
        
    uL, uR = Function(V), Function(V)
    uL.vector.array = left_bc
    uR.vector.array = right_bc
    
    bcL, bcR = dirichletbc(uL, dofs_L), dirichletbc(uR, dofs_R)
    
    v_norm = sqrt(dot(adv, adv))
    
    tao = Constant(mesh_1d, ScalarType(0.5*adv*beta*h/v_norm))
    
    adv = Constant(mesh_1d, ScalarType(adv))
    mu = Constant(mesh_1d, ScalarType(mu))
    sigma = Constant(mesh_1d, ScalarType(sigma))

    F = (w*adv*u.dx(0) + mu*w.dx(0)*u.dx(0) + sigma*u - w*s)*dx \
        + (adv*w.dx(0) - div(mu*grad(w)) + sigma*w)*tao*(adv*u.dx(0) - div(mu*grad(u)) + sigma*u - s)*dx
    # Eq. (2.61) (2.62) （2.57）

    a, L = lhs(F), rhs(F)

    problem = petsc.LinearProblem(a, L, bcs=[bcL, bcR])
    u = problem.solve()
    
    return u, V.tabulate_dof_coordinates().T[0]

def steady_adv_diff_1d_FV(mu, nx, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    # mesh of 10 uniform linear elements
    mesh_1d = create_interval(MPI.COMM_WORLD, nx, (0.0, 1.0))

    h = 1.0/nx
    adv = as_vector([1.0])
    eta = Constant(mesh_1d, ScalarType(1e5))
    one = Constant(mesh_1d, ScalarType(1.0))

    Pe = 1.0*h/(2*mu)
    print('Peclet number = ', Pe)

    V = FunctionSpace(mesh_1d, ('DG', 0))
    u = TrialFunction(V)
    w = TestFunction(V)
    
    n = FacetNormal(mesh_1d)
    
    x_ = Function(V)
    x_.interpolate(lambda x: x[0])
    Delta_h = sqrt(jump(x_)**2)
    
    boundaries = [(2, lambda x: np.isclose(x[0], 0)),
                  (3, lambda x: np.isclose(x[0], 1)),]
    
    # Start of Scripts by Jørgen S. Dokken, FEniCSx tutorial
    
    facet_indices, facet_markers = [], []
    fdim = mesh_1d.topology.dim - 1
    cdim = mesh_1d.topology.dim
    
    all_facet_indices = locate_entities(mesh_1d, fdim, lambda x: x[0] < np.inf)
    all_cell_indices = locate_entities(mesh_1d, cdim, lambda x: x[0] < np.inf)
    
    for (marker, locator) in boundaries:
        facets = locate_entities(mesh_1d, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)    
   
    all_facet_tag = meshtags(mesh_1d, fdim, all_facet_indices, np.zeros_like(all_facet_indices).astype(np.int32))
    cell_tag = meshtags(mesh_1d, cdim, all_cell_indices, np.zeros_like(all_cell_indices).astype(np.int32))
    facet_tag = meshtags(mesh_1d, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    
    # End of Scripts by Jørgen S. Dokken, FEniCSx tutorial
        
    dS = Measure('dS', domain=mesh_1d, subdomain_data=all_facet_tag)
    ds = Measure('ds', domain=mesh_1d, subdomain_data=facet_tag)
    dx = Measure('dx', domain=mesh_1d, subdomain_data=cell_tag)

    adv_np = ( dot ( adv, n ) + Abs ( dot ( adv, n ) ) ) / 2.0
    adv_nm = ( dot ( adv, n ) - Abs ( dot ( adv, n ) ) ) / 2.0
    
    F = mu*jump(w)*jump(u)/Delta_h*dS - w*s*dx\
        - w*(left_bc - u)/(x_ - 0.0)*ds(2) - w*(right_bc - u)/(one -x_)*ds(3) \
        + jump(w)*(adv_np('+')*u('+') - adv_np('-')*u('-'))*dS(0)\
        
    a, L = lhs(F), rhs(F)

    problem = petsc.LinearProblem(a, L, bcs=[])
    u = problem.solve()
    
    return u, V.tabulate_dof_coordinates().T[0]