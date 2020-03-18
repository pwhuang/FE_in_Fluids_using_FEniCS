import numpy as np
from dolfin import *

def steady_adv_diff_1d(mu, order, nx, s, left_bc, right_bc):
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
    #s = Constant(1.0)

    F = Constant(adv)*w*u.dx(0)*dx + Constant(mu)*w.dx(0)*u.dx(0)*dx - w*s*dx

    a, L = lhs(F), rhs(F)

    u = Function(V)
    solve(a==L, u, bcs=bc)
    
    return u, V.dofmap().dofs(), V.tabulate_dof_coordinates().T[0] #vertex_to_dof_map(V)

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