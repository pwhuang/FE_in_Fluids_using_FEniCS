import numpy as np
from dolfin import *

def transient_adv_diff_1d(mu, order, nx, s, left_bc, right_bc, dt_num, steps, theta_num):
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
    CFL = adv*dt_num/h
    
    print('Peclet number = ', Pe)
    print('Courant number = ', CFL)

    V = FunctionSpace(mesh_1d, 'CG', order)
    u = TrialFunction(V)
    w = TestFunction(V)
    u0 = Function(V)
    
    u_list = []

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc_L = DirichletBC(V, Constant(left_bc), boundary_L)
    bc_R = DirichletBC(V, Constant(right_bc), boundary_R)

    bc = [bc_L, bc_R]
    
    dt = Constant(dt_num)
    theta = Constant(theta_num)
    one = Constant(1.0)
    
    a = ( w*u/dt + theta*(w*Constant(adv)*u.dx(0) + Constant(mu)*inner(grad(w), grad(u))) )*dx
    L = ( w*u0/dt + w*s - (one-theta)*(w*Constant(adv)*u0.dx(0) + Constant(mu)*inner(grad(w), grad(u0))) )*dx

    #F = Constant(adv)*w*u.dx(0)*dx + Constant(mu)*w.dx(0)*u.dx(0)*dx - w*s*dx

    #a, L = lhs(F), rhs(F)

    u = Function(V)
    
    for i in range(steps):
        solve(a==L, u, bcs=bc)
        u0.assign(u)
        u_list.append(u0.copy())

    return u_list, V.dofmap().dofs(), V.tabulate_dof_coordinates().T[0] #vertex_to_dof_map(V)

def transient_adv_diff_1d_R11(mu, order, nx, s, left_bc, right_bc, dt_num, steps, theta_num):
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
    CFL = adv*dt_num/h
    
    print('Peclet number = ', Pe)
    print('Courant number = ', CFL)

    V = FunctionSpace(mesh_1d, 'CG', order)
    u = TrialFunction(V)
    w = TestFunction(V)
    u0 = Function(V)
    
    u_list = []

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc_L = DirichletBC(V, Constant(left_bc), boundary_L)
    bc_R = DirichletBC(V, Constant(right_bc), boundary_R)

    bc = [bc_L, bc_R]
    
    dt = Constant(dt_num)
    theta = Constant(theta_num)
    one = Constant(1.0)
    
    #a = ( w*u/dt + theta*(w*Constant(adv)*u.dx(0) + Constant(mu)*inner(grad(w), grad(u))) )*dx
    #L = ( w*u0/dt + w*s - (one-theta)*(w*Constant(adv)*u0.dx(0) + Constant(mu)*inner(grad(w), grad(u0))) )*dx
    
    def L(w, u):
        return w*Constant(adv)*u.dx(0) + Constant(mu)*inner(grad(w), grad(u))
    
    delta_u = u - u0
    W = 0.5
    ww = (w*s-L(w, u0))
    LL = L(w, u - u0)
    
    
    F = (w*delta_u/dt + W*LL - ww)*dx

    a, L = lhs(F), rhs(F)

    u = Function(V)
    
    for i in range(steps):
        solve(a==L, u, bcs=bc)
        u0.assign(u)
        u_list.append(u0.copy())

    return u_list, V.dofmap().dofs(), V.tabulate_dof_coordinates().T[0] #vertex_to_dof_map(V)

def transient_adv_diff_1d_R22(mu, order, nx, s, left_bc, right_bc, dt_num, steps, theta_num):
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
    CFL = adv*dt_num/h
    
    print('Peclet number = ', Pe)
    print('Courant number = ', CFL)
    
    P1 = FiniteElement("P", mesh_1d.ufl_cell(), 1)
    TH = MixedElement([P1, P1])
    V = FunctionSpace(mesh_1d, TH)
    V0 = V.sub(0).collapse()
    u = TrialFunction(V)
    w = TestFunction(V)
    u0 = Function(V0)
    
    u_list = []

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc_L = DirichletBC(V, (Constant(left_bc), Constant(left_bc)), boundary_L)
    bc_R = DirichletBC(V, (Constant(right_bc), Constant(right_bc)), boundary_R)

    bc = [bc_L, bc_R]
    
    dt = Constant(dt_num)
    theta = Constant(theta_num)
    one = Constant(1.0)
    
    def L(w, u):
        return w*Constant(adv)*u.dx(0) + Constant(mu)*inner(grad(w), grad(u))
    
    delta_u = as_vector([w[0]*(u[0] - u0), w[1]*(u[1] - u[0])])
    W = as_matrix([[7.0/24, -1.0/24], [13.0/24, 5.0/24]])
    #ww = as_vector( [0.5*(w[0]*s-L(w[0], u[1])), 0.5*(w[1]*s-L(w[1], u0))] )
    LL = as_vector([L(w[0], u[0] - u0), L(w[1], u[1] - u[0])])
    
    #a = ( w*u/dt + theta*L(w, u) )*dx
    #L = ( w*u0/dt + w*s - (one-theta)*L(w, u0) )*dx

    #F = dot( delta_u/dt + W*LL - ww, as_vector([1.0, 1.0]) )*dx
    F = dot( delta_u/dt + W*LL, as_vector([1.0, 1.0]) )*dx - (0.5*(w[0]*s-L(w[0], u0)) + 0.5*(w[1]*s-L(w[1], u0)))*dx
    
    #F = 24*w[0]*(u[0] - u0[0])/dt*dx - (-L(w[0], u[1]) + 8*L(v[0], u[0]) + 5*L(v[0], u0[0])) \
    #    +24*w[1]*(u[1] - u[0])/dt*dx - ( 5*L(w[1], u[1]) + 8*L(v[1], u[0]) - L(v[1], u0[1]))

    a, L = lhs(F), rhs(F)

    u = Function(V)
    
    for i in range(steps):
        solve(a==L, u, bcs=bc)
        
        u1, u2 = u.split(True)
        u0.assign(u2)
        u_list.append(u0.copy())

    return u_list, V.dofmap().dofs(), V.sub(0).collapse().tabulate_dof_coordinates().T[0] #vertex_to_dof_map(V)
