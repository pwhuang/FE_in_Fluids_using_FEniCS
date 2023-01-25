import numpy as np
from dolfinx.fem import *
from dolfinx.mesh import create_interval, exterior_facet_indices
from ufl import TrialFunction, TestFunction, dx, ds, dS, dot, inner, lhs, rhs
from ufl.algebra import Abs
from mpi4py import MPI

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

def steady_adv_diff_1d_SU(mu, order, nx, method, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    # mesh of 10 uniform linear elements
    mesh_1d = IntervalMesh(nx, 0, 1)

    h = 1.0/nx
    adv = 1.0
    
    Pe = adv*h/(2*mu)
    
    if method=='full_upwind':
        beta = 1.0
    elif method=='optimal':
        beta = 1.0/np.tanh(Pe) - 1.0/Pe 
    
    print('Peclet number = ', Pe)

    V = FunctionSpace(mesh_1d, 'CG', order)
    u = TrialFunction(V)
    w = TestFunction(V)

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc_L = DirichletBC(V, Constant(left_bc), boundary_L)
    bc_R = DirichletBC(V, Constant(right_bc), boundary_R)

    bc = [bc_L, bc_R]

    # Define Source term
    
    #s = Constant(1)
    
    F = (w + Constant(beta*h/2)*w.dx(0) )*Constant(adv)*u.dx(0)*dx + Constant(mu)*w.dx(0)*u.dx(0)*dx - w*s*dx

    a, L = lhs(F), rhs(F)

    u = Function(V)
    solve(a==L, u, bcs=bc)
    
    return u, V.dofmap().dofs(), V.tabulate_dof_coordinates().T[0]

def steady_adv_diff_1d_SUPG(mu, order, nx, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    # mesh of 10 uniform linear elements
    mesh_1d = IntervalMesh(nx, 0, 1)
    
    h = 1.0/nx/order
    adv = 1.0
    
    Pe = adv*h/(2*mu)
    beta = 1.0/np.tanh(Pe) - 1.0/Pe 
    
    print('Peclet number = ', Pe)

    V = FunctionSpace(mesh_1d, 'CG', order)
    u = TrialFunction(V)
    w = TestFunction(V)

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc_L = DirichletBC(V, Constant(left_bc), boundary_L)
    bc_R = DirichletBC(V, Constant(right_bc), boundary_R)

    bc = [bc_L, bc_R]
    
    v_norm = dot(adv, adv)
    
    tao = 0.5*adv*beta*h/v_norm # p. 61
    
    adv = Constant(adv)

    F = w*adv*u.dx(0)*dx + Constant(mu)*w.dx(0)*u.dx(0)*dx - w*s*dx \
        + adv*Constant(tao)*w.dx(0)*(adv*u.dx(0) - div(Constant(mu)*grad(u)) - s)*dx
    # Eq. (2.55) (2.56) （2.57）

    a, L = lhs(F), rhs(F)

    u = Function(V)
    solve(a==L, u, bcs=bc, solver_parameters={'linear_solver': 'gmres',\
                             'preconditioner': 'ilu'})
    
    return u, V.dofmap().dofs(), V.tabulate_dof_coordinates().T[0]

def steady_adv_diff_1d_GLS(mu, sigma, order, nx, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    # mesh of 10 uniform linear elements
    mesh_1d = IntervalMesh(nx, 0, 1)
    
    h = 1.0/nx
    adv = 1.0
    
    Pe = adv*h/(2*mu)
    beta = 1.0/np.tanh(Pe) - 1.0/Pe 
    
    print('Peclet number = ', Pe)

    V = FunctionSpace(mesh_1d, 'CG', order)
    u = TrialFunction(V)
    w = TestFunction(V)

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc_L = DirichletBC(V, Constant(left_bc), boundary_L)
    bc_R = DirichletBC(V, Constant(right_bc), boundary_R)

    bc = [bc_L, bc_R]
    #bc = []
    
    v_norm = sqrt(dot(adv, adv))
    
    tao = 0.5*adv*beta*h/v_norm
    
    adv = Constant(adv)
    mu = Constant(mu)
    sigma = Constant(sigma)

    F = (w*adv*u.dx(0) + mu*w.dx(0)*u.dx(0) + sigma*u - w*s)*dx \
        + (adv*w.dx(0) - div(mu*grad(w)) + sigma*w)*Constant(tao)*(adv*u.dx(0) - div(mu*grad(u)) + sigma*u - s)*dx
    # Eq. (2.61) (2.62) （2.57）

    a, L = lhs(F), rhs(F)

    u = Function(V)
    solve(a==L, u, bcs=bc, solver_parameters={'linear_solver': 'gmres',\
                             'preconditioner': 'ilu'})
    
    return u, V.dofmap().dofs(), V.tabulate_dof_coordinates().T[0]

def steady_adv_diff_1d_FV(mu, nx, s, left_bc, right_bc):
    # Input
    # mu:     diffusivity, constant
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    # mesh of 10 uniform linear elements
    mesh_1d = IntervalMesh(nx, 0, 1)

    h = 1.0/nx
    adv = as_vector([1.0])
    eta = Constant(1e5)
    one = Constant(1.0)

    Pe = 1.0*h/(2*mu)
    print('Peclet number = ', Pe)

    V = FunctionSpace(mesh_1d, 'DG', 0)
    u = TrialFunction(V)
    w = TestFunction(V)
    
    n = FacetNormal(mesh_1d)
    
    x_ = Function(V)
    x_.interpolate(lambda x: x[0])
    Delta_h = sqrt(jump(x_)**2)
    
    boundary_markers = MeshFunction('size_t', mesh_1d, dim=0)

    class left(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

    class right(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1.0, DOLFIN_EPS)
    
    b_left = left()
    b_right = right()
    
    boundary_markers.set_all(0)

    b_left.mark(boundary_markers, 2)
    b_right.mark(boundary_markers, 3)

    bc = []
    
    dS = Measure('dS', domain=mesh_1d, subdomain_data=boundary_markers)
    ds = Measure('ds', domain=mesh_1d, subdomain_data=boundary_markers)
    dx = Measure('dx', domain=mesh_1d, subdomain_data=boundary_markers)
    
    adv_np = ( dot ( adv, n ) + Abs ( dot ( adv, n ) ) ) / 2.0
    adv_nm = ( dot ( adv, n ) - Abs ( dot ( adv, n ) ) ) / 2.0
    #adv_n = dot ( adv, n )
    
    F = Constant(mu)*jump(w)*jump(u)/Delta_h*dS - w*s*dx\
        - w*(Constant(left_bc) - u)/(x_ - 0.0)*ds(2) - w*(Constant(right_bc) - u)/(one -x_)*ds(3) \
        + jump(w)*(adv_np('+')*u('+') - adv_np('-')*u('-'))*dS(0)\
        
    a, L = lhs(F), rhs(F)

    u = Function(V)
    solve(a==L, u, bcs=bc)
    
    return u, V.dofmap().dofs(), V.tabulate_dof_coordinates().T[0] #vertex_to_dof_map(V)