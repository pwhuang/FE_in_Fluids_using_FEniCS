import numpy as np
from dolfin import *
from ufl.algebra import Abs

def transient_adv_2d_R11(mesh_2d, s, init_cond, dt_num, steps, theta_num):
    # Input
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function

    V = FunctionSpace(mesh_2d, 'CG', 1)
    Vec = VectorFunctionSpace(mesh_2d, 'CG', 1)
    
    u = TrialFunction(V)
    w = TestFunction(V)
    u0 = project(init_cond, V)
    adv = project(Expression( ('sin(pi*x[0])*cos(pi*x[1])', '-cos(pi*x[0])*sin(pi*x[1])'), degree=1 ), Vec )
    
    u_list = [u0.copy()]

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    #bc_L = DirichletBC(V, Constant(left_bc), boundary_L)
    #bc_R = DirichletBC(V, Constant(right_bc), boundary_R)

    bc = []
    
    dt = Constant(dt_num)
    one = Constant(1.0)
    
    def L(w, u):
        return Constant(0.5)*w*div(adv*u) + Constant(0.5)*w*dot(adv, grad(u))
        #return w*dot(adv, grad(u))
        
    delta_u = u - u0
    W = 0.5
    ww = (w*s-L(w, u0))
    
    #F = (w*delta_u/dt + W*L(w, u - u0) - ww)*dx
        
    #a, L = lhs(F), rhs(F)
    
    theta = Constant(theta_num)
    
    a = ( w*u/dt + theta*L(w, u) )*dx
    L = ( w*u0/dt + w*s - (one-theta)*L(w, u0) )*dx

    u = Function(V)
    
    problem = LinearVariationalProblem(a, L, u, bcs=bc)
    solver = LinearVariationalSolver(problem)
    
    prm = solver.parameters
    
    prm['krylov_solver']['absolute_tolerance'] = 1e-12
    prm['krylov_solver']['relative_tolerance'] = 1e-10
    prm['krylov_solver']['maximum_iterations'] = 20
    #if iterative_solver:
    prm['linear_solver'] = 'gmres'
    prm['preconditioner'] = 'ilu'
    
    for i in range(steps):
        solver.solve()
        u0.assign(u)
        u_list.append(u0.copy())

    return u_list

def transient_adv_2d_R22(mesh_2d, s, init_cond, dt_num, steps):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    
    P1 = FiniteElement("P", mesh_2d.ufl_cell(), 1)
    TH = MixedElement([P1, P1])
    V = FunctionSpace(mesh_2d, TH)
    V0 = V.sub(0).collapse()
    u = TrialFunction(V)
    w = TestFunction(V)
    #u0 = Function(V0)
    u0 = project(init_cond, V0)
    
    Vec = VectorFunctionSpace(mesh_2d, 'CG', 1)
    
    adv = project(Expression( ('sin(pi*x[0])*cos(pi*x[1])', '-cos(pi*x[0])*sin(pi*x[1])'), degree=1 ), Vec )
    
    u_list = [u0.copy()]

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc = []
    
    dt = Constant(dt_num)
    one = Constant(1.0)
    
    def L(w, u):
        return Constant(0.5)*w*div(adv*u) + Constant(0.5)*w*dot(adv, grad(u))
        #return w*dot(adv, grad(u))
    
    delta_u = as_vector([(u[0] - u0), (u[1] - u[0])])
    W = as_matrix([[7.0/24, -1.0/24], [13.0/24, 5.0/24]])
    W_inv = inv(W)
    WU = W*delta_u
    #ww = as_vector( [0.5*(w[0]*s-L(w[0], u[1])), 0.5*(w[1]*s-L(w[1], u0))] )
    LL = as_vector([L(w[0], u[0] - u0), L(w[1], u[1] - u[0])])
    ww = as_vector([0.5, 0.5])
    
    
    F = dot( delta_u/dt, as_vector([w[0], w[1]]) )*dx\
        + (L(w[0], WU[0]) + L(w[1], WU[1]))*dx \
        + 0.5*(L(w[0], u0) + L(w[1], u0))*dx \
        - (0.5*(w[0]*s) + 0.5*(w[1]*s))*dx

    
    a, L = lhs(F), rhs(F)

    u = Function(V)
    
    problem = LinearVariationalProblem(a, L, u, bcs=bc)
    solver = LinearVariationalSolver(problem)
    
    prm = solver.parameters
    
    prm['krylov_solver']['absolute_tolerance'] = 1e-12
    prm['krylov_solver']['relative_tolerance'] = 1e-10
    prm['krylov_solver']['maximum_iterations'] = 20
    #if iterative_solver:
    prm['linear_solver'] = 'gmres'
    prm['preconditioner'] = 'ilu'
    
    for i in range(steps):
        solver.solve()
        u1, u2 = u.split(True)
        u0.assign(u2)
        u_list.append(u0.copy())

    return u_list

def transient_adv_2d_upwind(mesh_2d, s, init_cond, dt_num, steps, theta_num, add_diff):
    # Input
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function

    V = FunctionSpace(mesh_2d, 'DG', 0)
    Vec = VectorFunctionSpace(mesh_2d, 'CR', 1)
    
    u = TrialFunction(V)
    w = TestFunction(V)
    u0 = project(init_cond, V)
    adv = project(Expression( ('sin(pi*x[0])*cos(pi*x[1])', '-cos(pi*x[0])*sin(pi*x[1])'), degree=1 ), Vec )
    
    n = FacetNormal(mesh_2d)
    
    x_ = interpolate(Expression("x[0]", degree=1), V)
    y_ = interpolate(Expression("x[1]",degree=1), V)
    Delta_h = sqrt(jump(x_)**2 + jump(y_)**2)
    
    u_list = [u0.copy()]

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    #bc_L = DirichletBC(V, Constant(left_bc), boundary_L)
    #bc_R = DirichletBC(V, Constant(right_bc), boundary_R)

    bc = []
    
    dt = Constant(dt_num)
    one = Constant(1.0)
    
    adv_np = ( dot ( adv, n ) + Abs ( dot ( adv, n ) ) ) / 2.0
    adv_nm = ( dot ( adv, n ) - Abs ( dot ( adv, n ) ) ) / 2.0
    adv_n = dot ( adv, n )
    
    def L(w, u):
        return dot(jump(w), adv_np('+')*u('+') + adv_nm('+')*u('-') )
               #- Constant(add_diff)*0.5*dt*jump(dot(adv, adv))*dot(jump(w, n), jump(u, n))/Delta_h
        #return dot(jump(w), adv_np('+')*u('+') - adv_np('-')*u('-') )
        #return Constant(0.5)*w*div(adv*u) + Constant(0.5)*w*dot(adv, grad(u))
        
        
    delta_u = u - u0
    W = 0.5
    ww = (w*s-L(w, u0))
    
    #F = (w*delta_u/dt + W*L(w, u - u0) - ww)*dx
        
    #a, L = lhs(F), rhs(F)
    
    theta = Constant(theta_num)
    
    a = ( w*u/dt )*dx + theta*L(w, u)*dS
    L = ( w*u0/dt + w*s )*dx - (one-theta)*L(w, u0)*dS

    u = Function(V)
    
    problem = LinearVariationalProblem(a, L, u, bcs=bc)
    solver = LinearVariationalSolver(problem)
    
    prm = solver.parameters
    
    prm['krylov_solver']['absolute_tolerance'] = 1e-12
    prm['krylov_solver']['relative_tolerance'] = 1e-10
    prm['krylov_solver']['maximum_iterations'] = 20
    #if iterative_solver:
    prm['linear_solver'] = 'gmres'
    prm['preconditioner'] = 'ilu'
    
    for i in range(steps):
        solver.solve()
        u0.assign(u)
        u_list.append(u0.copy())

    return u_list

def transient_adv_2d_R22_FV(mesh_2d, s, init_cond, dt_num, steps):
    # Input
    # mu:     diffusivity, constant
    # order:  element order. 1 = linear elements. 2 = quadratic elements.
    # nx:     number of elements
    
    # Output
    # u:      the solution, dolfin function
    
    
    P1 = FiniteElement("DG", mesh_2d.ufl_cell(), 0)
    TH = MixedElement([P1, P1])
    V = FunctionSpace(mesh_2d, TH)
    V0 = V.sub(0).collapse()
    u = TrialFunction(V)
    w = TestFunction(V)
    #u0 = Function(V0)
    u0 = project(init_cond, V0)
    
    DG_space = FunctionSpace(mesh_2d, 'DG', 0)
    Vec = VectorFunctionSpace(mesh_2d, 'CR', 1)
    
    adv = project(Expression( ('sin(pi*x[0])*cos(pi*x[1])', '-cos(pi*x[0])*sin(pi*x[1])'), degree=1 ), Vec )
    
    n = FacetNormal(mesh_2d)
    
    x_ = interpolate(Expression("x[0]", degree=1), DG_space)
    y_ = interpolate(Expression("x[1]",degree=1), DG_space)
    Delta_h = sqrt(jump(x_)**2 + jump(y_)**2)
    
    u_list = [u0.copy()]

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1, DOLFIN_EPS)

    bc = []
    
    dt = Constant(dt_num)
    one = Constant(1.0)
    
    adv_np = ( dot ( adv, n ) + Abs ( dot ( adv, n ) ) ) / 2.0
    adv_nm = ( dot ( adv, n ) - Abs ( dot ( adv, n ) ) ) / 2.0
    #adv_n = dot ( adv, n )
    
    def L(w, u):
        #return Constant(0.5)*dot(jump(w), adv_np('+')*u('+') + adv_nm('+')*u('-') )\
        #     + Constant(0.5)*jump(w)*avg(dot( adv, n ))*jump(u)/Delta_h
        #return dot(w('-'), (dot(adv_n, u('-') - u('+') )))
        return dot(jump(w), adv_np('+')*u('+') - adv_np('-')*u('-') )
    
    delta_u = as_vector([(u[0] - u0), (u[1] - u[0])])
    W = as_matrix([[7.0/24, -1.0/24], [13.0/24, 5.0/24]])
    W_inv = inv(W)
    WU = W*delta_u
    #ww = as_vector( [0.5*(w[0]*s-L(w[0], u[1])), 0.5*(w[1]*s-L(w[1], u0))] )
    LL = as_vector([L(w[0], u[0] - u0), L(w[1], u[1] - u[0])])
    ww = as_vector([0.5, 0.5])
    
    
    F = dot( delta_u/dt, as_vector([w[0], w[1]]) )*dx\
        + (L(w[0], WU[0]) + L(w[1], WU[1]))*dS \
        + 0.5*(L(w[0], u0) + L(w[1], u0))*dS \
        - (0.5*(w[0]*s) + 0.5*(w[1]*s))*dx

    
    a, L = lhs(F), rhs(F)

    u = Function(V)
    
    problem = LinearVariationalProblem(a, L, u, bcs=bc)
    solver = LinearVariationalSolver(problem)
    
    prm = solver.parameters
    
    prm['krylov_solver']['absolute_tolerance'] = 1e-12
    prm['krylov_solver']['relative_tolerance'] = 1e-10
    prm['krylov_solver']['maximum_iterations'] = 1000
    #if iterative_solver:
    prm['linear_solver'] = 'gmres'
    prm['preconditioner'] = 'ilu'
    
    for i in range(steps):
        solver.solve()
        
        u1, u2 = u.split(True)
        u0.assign(u2)
        u_list.append(u0.copy())

    return u_list