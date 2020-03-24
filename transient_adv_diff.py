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

def transient_adv_diff_1d_R11(mu, order, nx, s, left_bc, right_bc, init_cond, dt_num, steps, method):
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
    #u0 = Function(V)
    u0 = project(init_cond, V)
    
    
    u_list = []

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc_L = DirichletBC(V, Constant(left_bc), boundary_L)
    bc_R = DirichletBC(V, Constant(right_bc), boundary_R)

    bc = [bc_L, bc_R]
    
    dt = Constant(dt_num)
    one = Constant(1.0)
    
    def L(w, u):
        return w*Constant(adv)*u.dx(0) + Constant(mu)*inner(grad(w), grad(u))
    
    delta_u = u - u0
    W = 0.5
    ww = (w*s-L(w, u0))
    
    # The residual term used for stabilization
    def R(u, u0):
        return delta_u/dt + W*Constant(adv)*u.dx(0) - (s - Constant(adv)*u0.dx(0))
    
    # the stabilization constant. p. 232, Soulaimani and Fortin (1994), Codina (2000)
    tau = Constant(1.0/(2.0/dt_num + 2.0*adv/h + 4.0*mu/h**2))
    
    if method=='Galerkin':
        def P(w):
            return Constant(0.0)
        
        F = (w*delta_u/dt + W*L(w, u - u0) - ww)*dx\
        + tau*P(w)*R(u, u0)*dx
    
    elif method=='SUPG':
        def P(w):
            return W*Constant(adv)*w.dx(0)
        
        F = (w*delta_u/dt + W*L(w, u - u0) - ww)*dx\
        + tau*P(w)*R(u, u0)*dx
    
    elif method=='GLS':
        def P(w):
            return w/dt + W*Constant(adv)*w.dx(0)
        
        F = (w*delta_u/dt + W*L(w, u - u0) - ww)*dx\
        + tau*P(w)*R(u, u0)*dx
    
    # This is a wrong implementation.
    elif method=='LS':
        def P(w):
            return W*Constant(adv)*w.dx(0)
        
        tau = dt
        
    
        F = (w*delta_u/dt + W*L(w, u - u0) - ww)*dx\
            + tau*P(w)*R(u, u0)*dx + w*R(u, u0)*dx
        
        
    a, L = lhs(F), rhs(F)

    u = Function(V)
    
    for i in range(steps):
        solve(a==L, u, bcs=bc)
        u0.assign(u)
        u_list.append(u0.copy())

    return u_list, V.dofmap().dofs(), V.tabulate_dof_coordinates().T[0] #vertex_to_dof_map(V)

def transient_adv_diff_1d_R22(mu, order, nx, s, left_bc, right_bc, init_cond, dt_num, steps, method):
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
    #u0 = Function(V0)
    u0 = project(init_cond, V0)
    
    u_list = []

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc_L = DirichletBC(V, (Constant(left_bc), Constant(left_bc)), boundary_L)
    bc_R = DirichletBC(V, (Constant(right_bc), Constant(right_bc)), boundary_R)

    bc = [bc_L, bc_R]
    
    dt = Constant(dt_num)
    one = Constant(1.0)
    
    def L(w, u):
        return w*Constant(adv)*u.dx(0) + Constant(mu)*inner(grad(w), grad(u))
    
    delta_u = as_vector([(u[0] - u0), (u[1] - u[0])])
    W = as_matrix([[7.0/24, -1.0/24], [13.0/24, 5.0/24]])
    W_inv = inv(W)
    WU = W*delta_u
    #ww = as_vector( [0.5*(w[0]*s-L(w[0], u[1])), 0.5*(w[1]*s-L(w[1], u0))] )
    LL = as_vector([L(w[0], u[0] - u0), L(w[1], u[1] - u[0])])
    ww = as_vector([0.5, 0.5])
    
    # The residual term used for stabilization
    def R(u, u0):
        return delta_u/dt + W*Constant(adv)*delta_u.dx(0) - ww*(s - Constant(adv)*u0.dx(0))
    
    # the stabilization constant. p. 232, Soulaimani and Fortin (1994), Codina (2000)
    tau = Constant(1.0/(2.0/dt_num + 2.0*adv/h + 4.0*mu/h**2))
    
    if method=='Galerkin':
        F = dot( delta_u/dt, as_vector([w[0], w[1]]) )*dx\
            + (L(w[0], WU[0]) + L(w[1], WU[1]))*dx \
            + 0.5*(L(w[0], u0) + L(w[1], u0))*dx \
            - (0.5*(w[0]*s) + 0.5*(w[1]*s))*dx
        
        #+ (w[0]*Constant(adv)*WU[0].dx(0) + w[1]*Constant(adv)*WU[1].dx(0) )*dx\
        #+ Constant(mu)*inner(grad(w[0]), grad(WU[0]))*dx\
        #+ Constant(mu)*inner(grad(w[1]), grad(WU[1]))*dx\
        
        #+ (0.5*w[0]*Constant(adv)*u0.dx(0) + 0.5*w[1]*Constant(adv)*u0.dx(0) )*dx\
        #+ 0.5*Constant(mu)*inner(grad(w[0]), grad(u0))*dx\
        #+ 0.5*Constant(mu)*inner(grad(w[1]), grad(u0))*dx\
        
    elif method=='SUPG':
        def P(w):
            return W*Constant(adv)*w.dx(0)
        
        tau = inv(W_inv/dt + Constant(2*adv/h + 4*mu/h**2)*Identity(2)).T * W_inv
        
        tauPw = tau*P(w)
        
        
        F = dot( delta_u/dt, as_vector([w[0], w[1]]) )*dx \
            + (L(w[0], WU[0]) + L(w[1], WU[1]))*dx \
            + 0.5*(L(w[0], u0) + L(w[1], u0))*dx \
            - (0.5*(w[0]*s) + 0.5*(w[1]*s))*dx \
            + dot(tauPw[0], R(u, u0)[0])*dx \
            + dot(tauPw[1], R(u, u0)[1])*dx
        
    
    
    a, L = lhs(F), rhs(F)

    u = Function(V)
    
    for i in range(steps):
        solve(a==L, u, bcs=bc)
        
        u1, u2 = u.split(True)
        u0.assign(u2)
        u_list.append(u0.copy())

    return u_list, V.dofmap().dofs(), V.sub(0).collapse().tabulate_dof_coordinates().T[0] #vertex_to_dof_map(V)
