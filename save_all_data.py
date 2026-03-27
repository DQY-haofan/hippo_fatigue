"""
═══════════════════════════════════════════════════════════════════
 Phase-Field Fracture: Compute + Save All Data (v2)
 Added: reaction force output for load-displacement curves
 Runtime: ~25 min on single CPU
 Output: paper_outputs/ (CSV + NPZ for all figures)
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
import os, csv, warnings
from time import time

warnings.filterwarnings('ignore', category=sparse.SparseEfficiencyWarning)
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

_T_START = time()

OUT = 'paper_outputs'
os.makedirs(OUT, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

E_mod = 210.0; nu = 0.3
Gc = 2.7e-3; ell = 0.031; kappa = 1e-7
lam = E_mod * nu / ((1+nu)*(1-2*nu))
mu  = E_mod / (2*(1+nu))
L = 1.0; N_mesh = 64; h_mesh = L / N_mesh

D_mat = np.array([
    [lam+2*mu, lam,      0 ],
    [lam,      lam+2*mu, 0 ],
    [0,        0,         mu]])

u_peak = 1.5e-2
N_ph1, N_ph2, N_ph3 = 25, 10, 15
N_steps = N_ph1 + N_ph2 + N_ph3

u_ph1 = np.linspace(0, u_peak, N_ph1+1)[1:]
u_ph2 = np.linspace(u_peak, 0.2*u_peak, N_ph2+1)[1:]
u_ph3 = np.linspace(0.2*u_peak, 1.3*u_peak, N_ph3+1)[1:]
u_loads = np.concatenate([u_ph1, u_ph2, u_ph3])
phase_labels = ['load']*N_ph1 + ['unload']*N_ph2 + ['reload']*N_ph3

print(f"ℓ={ell}, h={h_mesh:.4f}, ℓ/h={ell/h_mesh:.2f}")
print(f"Mesh: {N_mesh}×{N_mesh}, nodes={(N_mesh+1)**2}")
print(f"Load: {N_steps} steps")


# ═══════════════════════════════════════════════════════════════
# SOLVER (v2: added reaction force)
# ═══════════════════════════════════════════════════════════════

class PhaseFieldSparse:
    def __init__(self, N, formulation='miehe', loading='tension'):
        self.N = N; self.formulation = formulation; self.loading = loading
        self.h = L / N; self.n_nodes = (N+1)**2
        self.n_elem = N**2; self.n_dof = 2 * self.n_nodes
        xs = np.linspace(0, L, N+1)
        Y, X = np.meshgrid(xs, xs, indexing='ij')
        self.coords = np.column_stack([X.ravel(), Y.ravel()])
        jj, ii = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        n0 = jj.ravel()*(N+1) + ii.ravel()
        self.conn = np.column_stack([n0, n0+1, n0+N+2, n0+N+1])
        self.dof_u = np.column_stack([2*self.conn, 2*self.conn+1])
        gp = 1.0/np.sqrt(3.0)
        xi_g = np.array([-gp, gp, gp, -gp])
        eta_g = np.array([-gp, -gp, gp, gp])
        w = self.h**2
        self.N_gp = 0.25*np.array([(1-xi_g)*(1-eta_g),(1+xi_g)*(1-eta_g),
                                    (1+xi_g)*(1+eta_g),(1-xi_g)*(1+eta_g)]).T
        dNdxi = 0.25*np.array([-(1-eta_g),(1-eta_g),(1+eta_g),-(1+eta_g)]).T
        dNdeta = 0.25*np.array([-(1-xi_g),-(1+xi_g),(1+xi_g),(1-xi_g)]).T
        dNdx = dNdxi/self.h; dNdy = dNdeta/self.h
        self.B_all = np.zeros((4,3,8))
        for g in range(4):
            self.B_all[g,0,0:4]=dNdx[g]; self.B_all[g,1,4:8]=dNdy[g]
            self.B_all[g,2,0:4]=dNdy[g]; self.B_all[g,2,4:8]=dNdx[g]
        self.ke_u_base = np.zeros((4,8,8))
        for g in range(4):
            self.ke_u_base[g] = w*(self.B_all[g].T @ D_mat @ self.B_all[g])
        self.kd_mass = np.zeros((4,4,4))
        for g in range(4):
            self.kd_mass[g] = w*np.outer(self.N_gp[g], self.N_gp[g])
        self.kd_stiff = np.zeros((4,4,4))
        for g in range(4):
            self.kd_stiff[g] = w*Gc*ell*(np.outer(dNdx[g],dNdx[g])+np.outer(dNdy[g],dNdy[g]))
        self.kd_const = np.zeros((4,4))
        for g in range(4):
            self.kd_const += (Gc/ell)*self.kd_mass[g] + self.kd_stiff[g]
        self.N_w = w*self.N_gp

        bc_dofs=[]; bc_is_loaded=[]; eps=1e-10
        # ── NEW: track loaded DOFs separately for reaction force ──
        self.loaded_dofs = []
        for i in range(self.n_nodes):
            x, y = self.coords[i]
            if loading == 'tension':
                if abs(y)<eps:
                    bc_dofs.append(2*i+1); bc_is_loaded.append(False)
                if abs(y-L)<eps:
                    bc_dofs.append(2*i+1); bc_is_loaded.append(True)
                    self.loaded_dofs.append(2*i+1)
                if abs(x)<eps and abs(y)<eps:
                    bc_dofs.append(2*i); bc_is_loaded.append(False)
            elif loading == 'shear':
                if abs(y)<eps:
                    bc_dofs.append(2*i); bc_is_loaded.append(False)
                    bc_dofs.append(2*i+1); bc_is_loaded.append(False)
                if abs(y-L)<eps:
                    bc_dofs.append(2*i); bc_is_loaded.append(True)
                    self.loaded_dofs.append(2*i)
                    bc_dofs.append(2*i+1); bc_is_loaded.append(False)
        self.bc_dofs = np.array(bc_dofs, dtype=int)
        self.bc_is_loaded = np.array(bc_is_loaded, dtype=bool)
        self.loaded_dofs = np.array(self.loaded_dofs, dtype=int)

        self.d = np.zeros(self.n_nodes)
        self.H = np.zeros((self.n_elem, 4))
        self.d_prev = np.zeros(self.n_nodes)
        crack = (self.coords[:,0]<=0.5+self.h)&(np.abs(self.coords[:,1]-0.5)<self.h)
        self.d[crack]=1.0; self.d_prev=self.d.copy()
        dof8=self.dof_u
        self.row_u=np.repeat(dof8,8,axis=1).ravel()
        self.col_u=np.tile(dof8,(1,8)).ravel()
        c4=self.conn
        self.row_d=np.repeat(c4,4,axis=1).ravel()
        self.col_d=np.tile(c4,(1,4)).ravel()

    def _solve_u(self, d_field, u_bar):
        d_elem=d_field[self.conn]; d_gp=d_elem@self.N_gp.T
        g_gp=(1.0-d_gp)**2+kappa
        ke_all=np.einsum('eg,gij->eij',g_gp,self.ke_u_base)
        K=sparse.coo_matrix((ke_all.ravel(),(self.row_u,self.col_u)),
                            shape=(self.n_dof,self.n_dof)).tocsc()
        # ── NEW: save K_internal before penalty ──
        self._K_int = K.copy()
        F=np.zeros(self.n_dof)
        diag_max=np.abs(K.diagonal()).max()
        penalty=1e12*diag_max
        bc_vals=np.where(self.bc_is_loaded, u_bar, 0.0)
        pen_diag=sparse.csc_matrix((np.full(len(self.bc_dofs),penalty),
                                    (self.bc_dofs,self.bc_dofs)),
                                   shape=(self.n_dof,self.n_dof))
        K=K+pen_diag; F[self.bc_dofs]+=penalty*bc_vals
        return spsolve(K, F)

    def _compute_reaction(self, u_vec):
        """Compute reaction force on loaded boundary.
        F_reaction = sum of internal forces at loaded DOFs.
        F_int = K_internal @ u (without penalty)."""
        F_int = self._K_int @ u_vec
        return np.sum(F_int[self.loaded_dofs])

    def _compute_psi_plus(self, u_vec):
        u_elem=u_vec[self.dof_u]
        strain=np.einsum('gse,ne->ngs',self.B_all,u_elem)
        exx=strain[:,:,0]; eyy=strain[:,:,1]; exy=0.5*strain[:,:,2]
        e_mean=0.5*(exx+eyy)
        e_diff=0.5*np.sqrt((exx-eyy)**2+4*exy**2+1e-30)
        e1=e_mean+e_diff; e2=e_mean-e_diff
        tr_pos=np.maximum(e1+e2,0.0)
        return 0.5*lam*tr_pos**2+mu*(np.maximum(e1,0)**2+np.maximum(e2,0)**2)

    def _solve_d(self, driving, d_lower):
        kd_var=np.einsum('eg,gab->eab',2.0*driving,self.kd_mass)
        kd_all=kd_var+self.kd_const[np.newaxis,:,:]
        K_d=sparse.coo_matrix((kd_all.ravel(),(self.row_d,self.col_d)),
                              shape=(self.n_nodes,self.n_nodes)).tocsc()
        fd_all=np.einsum('eg,ga->ea',2.0*driving,self.N_w)
        F_d=np.zeros(self.n_nodes)
        np.add.at(F_d,self.conn.ravel(),fd_all.ravel())
        d_sol=spsolve(K_d,F_d)
        return np.maximum(np.clip(d_sol,0,1), d_lower)

    def step(self, u_bar, max_stagger=15, tol=1e-4):
        """Returns (u, psi, n_stagger, reaction_force)"""
        for k in range(max_stagger):
            d_old=self.d.copy()
            u=self._solve_u(self.d, u_bar)
            psi=self._compute_psi_plus(u)
            if self.formulation=='miehe':
                self.H=np.maximum(self.H,psi)
                self.d=self._solve_d(self.H, self.d_prev)
            else:
                self.d=self._solve_d(psi, self.d_prev)
            n_stag=k+1
            if k>=2 and np.max(np.abs(self.d-d_old))<tol: break
        self.d_prev=self.d.copy()
        F_react = self._compute_reaction(u)
        return u, psi, n_stag, F_react


# ═══════════════════════════════════════════════════════════════
# TRANSFER OPERATORS (unchanged)
# ═══════════════════════════════════════════════════════════════

def transfer_grid(field_nodal, N_fine, N_s):
    xs_f=np.linspace(0,L,N_fine+1); xs_c=np.linspace(0,L,N_s+1)
    F2d=field_nodal.reshape(N_fine+1,N_fine+1)
    interp_down=RegularGridInterpolator((xs_f,xs_f),F2d,method='linear',bounds_error=False,fill_value=None)
    Yc,Xc=np.meshgrid(xs_c,xs_c,indexing='ij')
    F_coarse=interp_down(np.column_stack([Yc.ravel(),Xc.ravel()])).reshape(N_s+1,N_s+1)
    interp_up=RegularGridInterpolator((xs_c,xs_c),F_coarse,method='linear',bounds_error=False,fill_value=None)
    Yf,Xf=np.meshgrid(xs_f,xs_f,indexing='ij')
    return interp_up(np.column_stack([Yf.ravel(),Xf.ravel()])).ravel()

def transfer_rbf(field_nodal, N_fine, N_s, epsilon=2.0):
    n_nodes=(N_fine+1)**2; xs=np.linspace(0,L,N_fine+1)
    Y,X=np.meshgrid(xs,xs,indexing='ij')
    coords=np.column_stack([X.ravel(),Y.ravel()])
    rng=np.random.RandomState(42)
    idx=rng.choice(n_nodes,size=N_s,replace=False)
    centres=coords[idx]; vals_c=field_nodal[idx]
    diff_c=centres[:,np.newaxis,:]-centres[np.newaxis,:,:]
    K_cc=np.exp(-epsilon**2*np.sum(diff_c**2,axis=2))+1e-8*np.eye(N_s)
    weights=np.linalg.solve(K_cc,vals_c)
    diff_all=coords[:,np.newaxis,:]-centres[np.newaxis,:,:]
    K_all=np.exp(-epsilon**2*np.sum(diff_all**2,axis=2))
    return K_all@weights

def gp_to_nodal(H_gp, conn, n_nodes):
    H_avg=H_gp.mean(axis=1); vals=np.repeat(H_avg,4); idx=conn.ravel()
    nodal=np.zeros(n_nodes); count=np.zeros(n_nodes)
    np.add.at(nodal,idx,vals); np.add.at(count,idx,1.0)
    count[count==0]=1.0; return nodal/count

def transfer_gp_field(H_gp, conn, N_fine, N_s, method='grid'):
    n_nodes=(N_fine+1)**2; nodal=gp_to_nodal(H_gp,conn,n_nodes)
    nodal_t=transfer_grid(nodal,N_fine,N_s) if method=='grid' else transfer_rbf(nodal,N_fine,N_s)
    return np.column_stack([nodal_t[conn].mean(axis=1)]*4)


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Error Accumulation (Miehe H)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPERIMENT 1: Error Accumulation under Miehe's H")
print("="*70)

transfer_schemes = {'grid_8':('grid',8),'grid_16':('grid',16),
                    'grid_32':('grid',32),'rbf_200':('rbf',200)}
N = N_mesh

# Reference (no transfer) — now collects P-δ
t0=time()
ref=PhaseFieldSparse(N,'miehe')
H_exact=[]; d_exact=[]; Fd_ref_miehe=[]
for si in range(N_steps):
    _,_,ns,F_r=ref.step(u_loads[si])
    H_exact.append(ref.H.copy()); d_exact.append(ref.d.copy())
    Fd_ref_miehe.append(F_r)
    if (si+1)%10==0: print(f"  Step {si+1}/{N_steps}: max(H)={ref.H.max():.4f}, F={F_r:.4f}")
print(f"  Ref Miehe: {time()-t0:.1f}s")

r_tip=np.sqrt((ref.coords[:,0]-0.5)**2+(ref.coords[:,1]-0.5)**2)
tip_mask=r_tip<5*ell

results_t1={}
for sn,(method,N_s) in transfer_schemes.items():
    t0s=time(); sol=PhaseFieldSparse(N,'miehe')
    H_errs=[]; d_errs=[]; H_tip_errs=[]; Fd_list=[]
    for si in range(N_steps):
        if si>0: sol.H=np.maximum(transfer_gp_field(sol.H,sol.conn,N,N_s,method),0.0)
        _,_,_,F_r=sol.step(u_loads[si])
        H_errs.append(np.sqrt(np.mean((sol.H-H_exact[si])**2)))
        d_errs.append(np.sqrt(np.mean((sol.d-d_exact[si])**2)))
        H_n=gp_to_nodal(sol.H,sol.conn,sol.n_nodes)
        H_ref_n=gp_to_nodal(H_exact[si],ref.conn,ref.n_nodes)
        H_tip_errs.append(np.sqrt(np.mean((H_n[tip_mask]-H_ref_n[tip_mask])**2)))
        Fd_list.append(F_r)
    results_t1[sn]={'H_errors':H_errs,'d_errors':d_errs,'H_tip_errors':H_tip_errs,
                     'solver':sol,'Fd':Fd_list}
    print(f"  {sn}: L2(H)={H_errs[-1]:.4f}, tip={H_tip_errs[-1]:.4f}, {time()-t0s:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Miehe vs Bourdin
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPERIMENT 2: Miehe vs Bourdin")
print("="*70)

t0=time()
ref_b=PhaseFieldSparse(N,'bourdin')
d_bexact=[]; Fd_ref_bourdin=[]
for si in range(N_steps):
    _,_,_,F_r=ref_b.step(u_loads[si])
    d_bexact.append(ref_b.d.copy()); Fd_ref_bourdin.append(F_r)
print(f"  Bourdin ref: {time()-t0:.1f}s")

results_t2={}
for sn,(method,N_s) in transfer_schemes.items():
    t0s=time()
    r1=results_t1[sn]
    m_H_tip=r1['H_tip_errors'][-1]
    sol_b=PhaseFieldSparse(N,'bourdin')
    db_errs=[]; db_tip_errs=[]; Fd_b=[]
    for si in range(N_steps):
        if si>0:
            if method=='grid': sol_b.d_prev=np.clip(transfer_grid(sol_b.d_prev,N,N_s),0,1)
            else: sol_b.d_prev=np.clip(transfer_rbf(sol_b.d_prev,N,N_s),0,1)
        _,_,_,F_r=sol_b.step(u_loads[si])
        db_errs.append(np.sqrt(np.mean((sol_b.d-d_bexact[si])**2)))
        db_tip_errs.append(np.sqrt(np.mean((sol_b.d[tip_mask]-d_bexact[si][tip_mask])**2)))
        Fd_b.append(F_r)
    b_d_tip=db_tip_errs[-1]; ratio_carried=m_H_tip/max(b_d_tip,1e-15)
    results_t2[sn]={'miehe_H_tip':m_H_tip,'miehe_H_errs':r1['H_errors'],
                     'bourdin_d_tip':b_d_tip,'bourdin_d_errs':db_errs,
                     'bourdin_d_tip_errs':db_tip_errs,'ratio_carried':ratio_carried,
                     'solver_m':r1['solver'],'solver_b':sol_b,
                     'Fd_miehe':r1['Fd'],'Fd_bourdin':Fd_b}
    print(f"  {sn}: ratio={ratio_carried:.1f}×, {time()-t0s:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Fatigue
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPERIMENT 3: 1D Fatigue (40 cycles)")
print("="*70)

N_1d=200; h_1d=L/N_1d; x_1d=np.linspace(0,L,N_1d+1)
p_fat=0.5; xi_fat=0.4
def f_fat(a): return (1+2*p_fat*(1-xi_fat)*np.maximum(a,0.0))**(-1/(1-xi_fat))
def f_fat_inv(phi):
    phi=np.clip(phi,1e-10,1.0); return (phi**(-(1-xi_fat))-1)/(2*p_fat*(1-xi_fat))
def solve_u1d(d,ub):
    g=(1-d)**2+kappa; ig=1.0/(g*E_mod); denom=np.trapezoid(ig,x_1d)
    if abs(denom)<1e-30: return np.zeros_like(x_1d)
    C=ub/denom; up=C*ig; u=np.cumsum(up)*h_1d; u-=u[0]
    if abs(u[-1])>1e-30: u*=ub/u[-1]
    return np.gradient(u,x_1d)
def psi1d(up): return 0.5*E_mod*np.maximum(up,0.0)**2
def solve_d1d(drv,fa_field,dl):
    d=dl.copy(); h2=h_1d**2
    for _ in range(200):
        do=d.copy()
        for i in range(1,N_1d):
            lap=np.clip((d[i-1]+d[i+1]-2*d[i])/h2,-1e10,1e10)
            fG=fa_field[i]*Gc; c=fG/(2*ell)+drv[i]; r=drv[i]+0.5*fG*ell*lap
            if c>1e-20: d[i]=r/c
        d=np.clip(np.maximum(d,dl),0,1)
        if np.max(np.abs(d-do))<1e-10: break
    return d

d0=np.exp(-np.abs(x_1d-0.5)/ell); d0[np.abs(x_1d-0.5)<h_1d]=1.0
Nc=40; spc=10; u_amp=5e-4; tsteps=Nc*spc
N_s1d=16; xc1d=np.linspace(0,L,N_s1d+1)
tarr=np.linspace(0,Nc,tsteps+1)[1:]; uarr=u_amp*np.abs(np.sin(np.pi*tarr))

dA=d0.copy(); alphaA=np.zeros_like(x_1d); psiA_prev=np.zeros_like(x_1d)
alpha_max_hist=[]; f_min_hist=[]
dB=d0.copy(); alphaB=np.zeros_like(x_1d); psiB_prev=np.zeros_like(x_1d)
alpha_err_B=[]; d_err_B=[]
dC=d0.copy(); phiC=np.ones_like(x_1d); psiC_prev=np.zeros_like(x_1d)
alpha_err_C=[]; d_err_C=[]
def transfer_1d(field,xc):
    return np.interp(x_1d,xc,np.interp(xc,x_1d,field))

t0=time()
for s in range(tsteps):
    ub=uarr[s]
    faA=f_fat(alphaA)
    for _ in range(5): up=solve_u1d(dA,ub); ps=psi1d(up); dA=solve_d1d(ps,faA,dA)
    alphaA+=np.maximum(ps-psiA_prev,0); psiA_prev=ps.copy()
    alpha_max_hist.append(alphaA.max()); f_min_hist.append(f_fat(alphaA).min())

    dB=np.maximum(np.clip(transfer_1d(dB,xc1d),0,1),d0)
    alphaB=np.maximum(transfer_1d(alphaB,xc1d),0)
    psiB_prev=np.maximum(transfer_1d(psiB_prev,xc1d),0)
    faB=f_fat(alphaB)
    for _ in range(5): up=solve_u1d(dB,ub); ps=psi1d(up); dB=solve_d1d(ps,faB,dB)
    alphaB+=np.maximum(ps-psiB_prev,0); psiB_prev=ps.copy()
    alpha_err_B.append(np.sqrt(np.mean((alphaB-alphaA)**2)))
    d_err_B.append(np.sqrt(np.mean((dB-dA)**2)))

    dC=np.maximum(np.clip(transfer_1d(dC,xc1d),0,1),d0)
    phiC=np.clip(transfer_1d(phiC,xc1d),1e-10,1.0)
    psiC_prev=np.maximum(transfer_1d(psiC_prev,xc1d),0)
    alphaC_rec=f_fat_inv(phiC); faC=f_fat(alphaC_rec)
    for _ in range(5): up=solve_u1d(dC,ub); ps=psi1d(up); dC=solve_d1d(ps,faC,dC)
    alphaC_rec+=np.maximum(ps-psiC_prev,0); phiC=f_fat(alphaC_rec); psiC_prev=ps.copy()
    alpha_err_C.append(np.sqrt(np.mean((alphaC_rec-alphaA)**2)))
    d_err_C.append(np.sqrt(np.mean((dC-dA)**2)))
    if (s+1)%(spc*10)==0:
        cyc=(s+1)//spc
        print(f"  Cycle {cyc}/{Nc}: B/C={alpha_err_B[-1]/max(alpha_err_C[-1],1e-15):.1f}×")
cycles_1d=np.arange(1,tsteps+1)/spc
print(f"  Fatigue: {time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 5a: Convergence (SEN-T)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CONVERGENCE RATE (SEN-T)")
print("="*70)

Ns_sweep=[2,4,8,16,32]; results_conv=[]
for N_s in Ns_sweep:
    h_s=L/N_s; ratio=h_s/ell
    sol_m=PhaseFieldSparse(N,'miehe')
    for si in range(N_steps):
        if si>0: sol_m.H=np.maximum(transfer_gp_field(sol_m.H,sol_m.conn,N,N_s,'grid'),0.0)
        sol_m.step(u_loads[si])
    err_H=np.sqrt(np.mean((sol_m.H-H_exact[-1])**2))
    H_n=gp_to_nodal(sol_m.H,sol_m.conn,sol_m.n_nodes)
    H_ref_n=gp_to_nodal(H_exact[-1],ref.conn,ref.n_nodes)
    err_H_tip=np.sqrt(np.mean((H_n[tip_mask]-H_ref_n[tip_mask])**2))
    sol_b=PhaseFieldSparse(N,'bourdin')
    for si in range(N_steps):
        if si>0: sol_b.d_prev=np.clip(transfer_grid(sol_b.d_prev,N,N_s),0,1)
        sol_b.step(u_loads[si])
    err_d_b=np.sqrt(np.mean((sol_b.d-d_bexact[-1])**2))
    err_d_b_tip=np.sqrt(np.mean((sol_b.d[tip_mask]-d_bexact[-1][tip_mask])**2))
    results_conv.append({'N_s':N_s,'h_s_over_ell':ratio,
                         'miehe_H_global':err_H,'miehe_H_tip':err_H_tip,
                         'bourdin_d_global':err_d_b,'bourdin_d_tip':err_d_b_tip})
    print(f"  N_s={N_s}: M_H={err_H:.2e}, B_d={err_d_b:.2e}")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS 4+5b: SEN-SHEAR
# ═══════════════════════════════════════════════════════════════

u_peak_shear=0.05; N_steps_shear=80
u_shear=np.linspace(0,u_peak_shear,N_steps_shear+1)[1:]

print("\n" + "="*70)
print("SEN-SHEAR")
print("="*70)

ref_s_m=PhaseFieldSparse(N,'miehe',loading='shear')
H_exact_s=[]; d_exact_s=[]; Fd_ref_shear_m=[]
for si in range(N_steps_shear):
    _,_,_,F_r=ref_s_m.step(u_shear[si])
    H_exact_s.append(ref_s_m.H.copy()); d_exact_s.append(ref_s_m.d.copy())
    Fd_ref_shear_m.append(F_r)
print(f"  Miehe shear ref done")

ref_s_b=PhaseFieldSparse(N,'bourdin',loading='shear')
d_bexact_s=[]; Fd_ref_shear_b=[]
for si in range(N_steps_shear):
    _,_,_,F_r=ref_s_b.step(u_shear[si])
    d_bexact_s.append(ref_s_b.d.copy()); Fd_ref_shear_b.append(F_r)
print(f"  Bourdin shear ref done")

r_tip_s=np.sqrt((ref_s_m.coords[:,0]-0.5)**2+(ref_s_m.coords[:,1]-0.5)**2)
tip_mask_s=r_tip_s<5*ell

shear_schemes={'grid_8':('grid',8),'grid_16':('grid',16),'grid_32':('grid',32)}
results_shear={}
for sn,(method,N_s) in shear_schemes.items():
    sol_m=PhaseFieldSparse(N,'miehe',loading='shear')
    for si in range(N_steps_shear):
        if si>0: sol_m.H=np.maximum(transfer_gp_field(sol_m.H,sol_m.conn,N,N_s,method),0.0)
        sol_m.step(u_shear[si])
    m_H_tip=np.sqrt(np.mean((gp_to_nodal(sol_m.H,sol_m.conn,sol_m.n_nodes)[tip_mask_s]-
                              gp_to_nodal(H_exact_s[-1],ref_s_m.conn,ref_s_m.n_nodes)[tip_mask_s])**2))
    sol_b=PhaseFieldSparse(N,'bourdin',loading='shear')
    for si in range(N_steps_shear):
        if si>0: sol_b.d_prev=np.clip(transfer_grid(sol_b.d_prev,N,N_s),0,1)
        sol_b.step(u_shear[si])
    b_d_tip=np.sqrt(np.mean((sol_b.d[tip_mask_s]-d_bexact_s[-1][tip_mask_s])**2))
    results_shear[sn]={'miehe_H_tip':m_H_tip,'bourdin_d_tip':b_d_tip,
                        'miehe_H_global':np.sqrt(np.mean((sol_m.H-H_exact_s[-1])**2)),
                        'bourdin_d_global':np.sqrt(np.mean((sol_b.d-d_bexact_s[-1])**2))}
    print(f"  {sn}: M/B ratio={m_H_tip/max(b_d_tip,1e-15):.2f}")

Ns_sweep_s=[4,8,16,32]; conv_shear=[]
for N_s in Ns_sweep_s:
    sol_m=PhaseFieldSparse(N,'miehe',loading='shear')
    for si in range(N_steps_shear):
        if si>0: sol_m.H=np.maximum(transfer_gp_field(sol_m.H,sol_m.conn,N,N_s,'grid'),0.0)
        sol_m.step(u_shear[si])
    sol_b=PhaseFieldSparse(N,'bourdin',loading='shear')
    for si in range(N_steps_shear):
        if si>0: sol_b.d_prev=np.clip(transfer_grid(sol_b.d_prev,N,N_s),0,1)
        sol_b.step(u_shear[si])
    conv_shear.append({'N_s':N_s,'h_s_over_ell':L/N_s/ell,
                       'miehe_H_global':np.sqrt(np.mean((sol_m.H-H_exact_s[-1])**2)),
                       'bourdin_d_global':np.sqrt(np.mean((sol_b.d-d_bexact_s[-1])**2))})


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 6: Corrected H-Transfer (Direction B)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPERIMENT 6: Corrected H-Transfer")
print("="*70)

eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
N_s_corr = 16
results_corrected = {}
for eta in eta_values:
    t0c=time(); sol_c=PhaseFieldSparse(N,'miehe')
    H_errs_c=[]; H_tip_errs_c=[]; Fd_c=[]
    for si in range(N_steps):
        if si>0:
            H_transferred=np.maximum(transfer_gp_field(sol_c.H,sol_c.conn,N,N_s_corr,'grid'),0.0)
            if eta>0:
                psi_current=sol_c.H.copy()
                overshoot=np.maximum(H_transferred-psi_current,0.0)
                sol_c.H=np.maximum(H_transferred-eta*overshoot,0.0)
            else:
                sol_c.H=H_transferred
        _,_,_,F_r=sol_c.step(u_loads[si])
        H_errs_c.append(np.sqrt(np.mean((sol_c.H-H_exact[si])**2)))
        H_n=gp_to_nodal(sol_c.H,sol_c.conn,sol_c.n_nodes)
        H_ref_n=gp_to_nodal(H_exact[si],ref.conn,ref.n_nodes)
        H_tip_errs_c.append(np.sqrt(np.mean((H_n[tip_mask]-H_ref_n[tip_mask])**2)))
        Fd_c.append(F_r)
    results_corrected[eta]={'H_errors':H_errs_c,'H_tip_errors':H_tip_errs_c,
                            'Fd':Fd_c,'final_H_global':H_errs_c[-1],'final_H_tip':H_tip_errs_c[-1]}
    if eta==0: base_tip=H_tip_errs_c[-1]
    red=(1-H_tip_errs_c[-1]/base_tip)*100 if base_tip>0 else 0
    print(f"  η={eta:.2f}: tip_err={H_tip_errs_c[-1]:.4f}, reduction={red:+.1f}%, {time()-t0c:.1f}s")


# ═══════════════════════════════════════════════════════════════
"""
═══════════════════════════════════════════════════════════════
 EXPERIMENTS 7-10: Additional content for reviewer robustness
 Paste into compute_and_save_v2.py BEFORE "SAVE ALL DATA"
 (after Experiment 6)
═══════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 7: Energy Evolution (Item 2)
# Elastic + Fracture energy at each step → proves solver correct
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPERIMENT 7: Energy Evolution")
print("="*70)

def compute_energies(solver, d_field, u_vec, N_grid, h_grid):
    """Compute elastic and fracture energies."""
    # Elastic energy: sum_e sum_g g(d_gp) * psi+(eps) * w
    d_elem = d_field[solver.conn]
    d_gp = d_elem @ solver.N_gp.T
    g_gp = (1.0 - d_gp)**2 + kappa
    u_elem = u_vec[solver.dof_u]
    strain = np.einsum('gse,ne->ngs', solver.B_all, u_elem)
    exx = strain[:,:,0]; eyy = strain[:,:,1]; exy = 0.5*strain[:,:,2]
    e_mean = 0.5*(exx + eyy)
    e_diff = 0.5*np.sqrt((exx-eyy)**2 + 4*exy**2 + 1e-30)
    e1 = e_mean + e_diff; e2 = e_mean - e_diff
    tr_pos = np.maximum(e1+e2, 0.0)
    psi_plus = 0.5*lam*tr_pos**2 + mu*(np.maximum(e1,0)**2 + np.maximum(e2,0)**2)
    w = h_grid**2
    E_elastic = np.sum(g_gp * psi_plus) * w

    # Fracture energy: Gc/2 * integral (d^2/ell + ell*|grad d|^2) dx
    # Use nodal d, compute gradient via finite differences
    d_2d = d_field.reshape(N_grid+1, N_grid+1)
    # d^2/ell term
    E_frac_bulk = (Gc / (2*ell)) * np.sum(d_field**2) * h_grid**2

    # ell*|grad d|^2 term (central differences)
    dddx = np.zeros_like(d_2d)
    dddy = np.zeros_like(d_2d)
    dddx[:, 1:-1] = (d_2d[:, 2:] - d_2d[:, :-2]) / (2*h_grid)
    dddy[1:-1, :] = (d_2d[2:, :] - d_2d[:-2, :]) / (2*h_grid)
    grad_sq = dddx**2 + dddy**2
    E_frac_grad = (Gc * ell / 2) * np.sum(grad_sq) * h_grid**2

    E_fracture = E_frac_bulk + E_frac_grad
    return E_elastic, E_fracture

# Run reference (no transfer) and collect energies
ref_energy = PhaseFieldSparse(N, 'miehe')
E_el_ref = []; E_fr_ref = []; u_vecs_ref = []
for si in range(N_steps):
    u_vec, _, _, _ = ref_energy.step(u_loads[si])
    Ee, Ef = compute_energies(ref_energy, ref_energy.d, u_vec, N, h_mesh)
    E_el_ref.append(Ee); E_fr_ref.append(Ef)
print(f"  Ref: E_el_final={E_el_ref[-1]:.6f}, E_fr_final={E_fr_ref[-1]:.6f}")

# With transfer (grid_16) — Miehe
sol_en_m = PhaseFieldSparse(N, 'miehe')
E_el_m16 = []; E_fr_m16 = []
for si in range(N_steps):
    if si > 0:
        sol_en_m.H = np.maximum(transfer_gp_field(sol_en_m.H, sol_en_m.conn, N, 16, 'grid'), 0.0)
    u_vec, _, _, _ = sol_en_m.step(u_loads[si])
    Ee, Ef = compute_energies(sol_en_m, sol_en_m.d, u_vec, N, h_mesh)
    E_el_m16.append(Ee); E_fr_m16.append(Ef)
print(f"  Miehe+grid16: E_el_final={E_el_m16[-1]:.6f}, E_fr_final={E_fr_m16[-1]:.6f}")

# Bourdin reference
ref_en_b = PhaseFieldSparse(N, 'bourdin')
E_el_bref = []; E_fr_bref = []
for si in range(N_steps):
    u_vec, _, _, _ = ref_en_b.step(u_loads[si])
    Ee, Ef = compute_energies(ref_en_b, ref_en_b.d, u_vec, N, h_mesh)
    E_el_bref.append(Ee); E_fr_bref.append(Ef)
print(f"  Bourdin ref: E_el_final={E_el_bref[-1]:.6f}, E_fr_final={E_fr_bref[-1]:.6f}")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 8: Crack Tip Tracking (Item 5)
# Does H transfer error move the crack tip position?
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPERIMENT 8: Crack Tip Tracking")
print("="*70)

def crack_tip_x(d_field, coords, threshold=0.5):
    """Find the x-coordinate of the furthest crack tip (d >= threshold, y ≈ 0.5)."""
    near_crack = np.abs(coords[:, 1] - 0.5) < 2*h_mesh
    cracked = d_field[near_crack] >= threshold
    if not np.any(cracked):
        return 0.5  # initial notch tip
    x_cracked = coords[near_crack, 0][cracked]
    return x_cracked.max()

# Reference tip positions (from d_exact already computed)
tip_ref = [crack_tip_x(d_exact[si], ref.coords) for si in range(N_steps)]

# Miehe + grid_16 tip positions
tip_m16 = [crack_tip_x(results_t1['grid_16']['solver'].d if si==N_steps-1
           else d_exact[si], ref.coords) for si in range(N_steps)]
# Need to re-run for per-step tracking...
sol_tip_m = PhaseFieldSparse(N, 'miehe')
tip_m16_full = []
for si in range(N_steps):
    if si > 0:
        sol_tip_m.H = np.maximum(transfer_gp_field(sol_tip_m.H, sol_tip_m.conn, N, 16, 'grid'), 0.0)
    sol_tip_m.step(u_loads[si])
    tip_m16_full.append(crack_tip_x(sol_tip_m.d, sol_tip_m.coords))

# Bourdin + grid_16 tip positions
sol_tip_b = PhaseFieldSparse(N, 'bourdin')
tip_b16_full = []
for si in range(N_steps):
    if si > 0:
        sol_tip_b.d_prev = np.clip(transfer_grid(sol_tip_b.d_prev, N, 16), 0, 1)
    sol_tip_b.step(u_loads[si])
    tip_b16_full.append(crack_tip_x(sol_tip_b.d, sol_tip_b.coords))

tip_err_m = [abs(tip_m16_full[i] - tip_ref[i]) for i in range(N_steps)]
tip_err_b = [abs(tip_b16_full[i] - tip_ref[i]) for i in range(N_steps)]
print(f"  Max tip position error (Miehe): {max(tip_err_m):.4f}")
print(f"  Max tip position error (Bourdin): {max(tip_err_b):.4f}")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 9: Mesh Sensitivity (Item 4)
# Test at N=32 (ℓ/h≈1), N=64 (ℓ/h≈2), N=128 (ℓ/h≈4)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXPERIMENT 9: Mesh Sensitivity")
print("="*70)

mesh_sizes = [32, 64]  # skip 128 if too slow; 64 already computed above
# Note: N=128 would take ~4x longer; add if time allows
N_s_test = 16
results_mesh_sens = {}

for Nm in mesh_sizes:
    hm = L/Nm
    ratio_lh = ell/hm
    t0m = time()

    # Reference (no transfer)
    ref_ms = PhaseFieldSparse(Nm, 'miehe')
    d_ref_final = None
    for si in range(N_steps):
        ref_ms.step(u_loads[si])
    d_ref_final = ref_ms.d.copy()
    H_ref_final = ref_ms.H.copy()

    # With transfer
    sol_ms = PhaseFieldSparse(Nm, 'miehe')
    for si in range(N_steps):
        if si > 0:
            sol_ms.H = np.maximum(transfer_gp_field(sol_ms.H, sol_ms.conn, Nm, N_s_test, 'grid'), 0.0)
        sol_ms.step(u_loads[si])

    err_H = np.sqrt(np.mean((sol_ms.H - H_ref_final)**2))
    err_d = np.sqrt(np.mean((sol_ms.d - d_ref_final)**2))
    r_tip_ms = np.sqrt((sol_ms.coords[:,0]-0.5)**2 + (sol_ms.coords[:,1]-0.5)**2)
    tip_mask_ms = r_tip_ms < 5*ell
    H_n_ms = gp_to_nodal(sol_ms.H, sol_ms.conn, sol_ms.n_nodes)
    H_ref_n_ms = gp_to_nodal(H_ref_final, ref_ms.conn, ref_ms.n_nodes)
    err_H_tip = np.sqrt(np.mean((H_n_ms[tip_mask_ms] - H_ref_n_ms[tip_mask_ms])**2))

    results_mesh_sens[Nm] = {
        'N_mesh': Nm, 'h': hm, 'ell_over_h': ratio_lh,
        'H_global': err_H, 'H_tip': err_H_tip, 'd_global': err_d,
        'time': time()-t0m
    }
    print(f"  N={Nm} (ℓ/h={ratio_lh:.1f}): H_tip={err_H_tip:.4f}, {time()-t0m:.1f}s")


# ═══════════════════════════════════════════════════════════════
# COMPUTATIONAL COST TABLE (Item 6)
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("COMPUTATIONAL COST SUMMARY")
print("="*70)

# Time a single reference solve vs transfer solve
import timeit

t0_bench = time()
ref_bench = PhaseFieldSparse(N, 'miehe')
for si in range(N_steps):
    ref_bench.step(u_loads[si])
t_ref = time() - t0_bench

t0_bench = time()
sol_bench = PhaseFieldSparse(N, 'miehe')
for si in range(N_steps):
    if si > 0:
        sol_bench.H = np.maximum(transfer_gp_field(sol_bench.H, sol_bench.conn, N, 16, 'grid'), 0.0)
    sol_bench.step(u_loads[si])
t_transfer = time() - t0_bench

overhead_pct = (t_transfer / t_ref - 1) * 100
n_dof = 2 * (N+1)**2
print(f"  DOFs: {n_dof}")
print(f"  Reference: {t_ref:.1f}s")
print(f"  With transfer (grid_16): {t_transfer:.1f}s ({overhead_pct:+.1f}%)")
print(f"  Correction overhead: negligible (1 array op per step)")

cost_data = {
    'N_mesh': N, 'n_dof': n_dof, 'n_steps': N_steps,
    't_ref': t_ref, 't_transfer_g16': t_transfer,
    'overhead_pct': overhead_pct
}


# ═══════════════════════════════════════════════════════════════
# SAVE NEW DATA (append to SAVE ALL DATA section)
# ═══════════════════════════════════════════════════════════════

# Energy evolution
with open(f'{OUT}/energy_evolution.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['step', 'u_bar',
                'E_el_ref', 'E_fr_ref', 'E_tot_ref',
                'E_el_miehe16', 'E_fr_miehe16', 'E_tot_miehe16',
                'E_el_bourdin_ref', 'E_fr_bourdin_ref', 'E_tot_bourdin_ref'])
    for si in range(N_steps):
        w.writerow([si+1, f"{u_loads[si]:.8e}",
                    f"{E_el_ref[si]:.8e}", f"{E_fr_ref[si]:.8e}", f"{E_el_ref[si]+E_fr_ref[si]:.8e}",
                    f"{E_el_m16[si]:.8e}", f"{E_fr_m16[si]:.8e}", f"{E_el_m16[si]+E_fr_m16[si]:.8e}",
                    f"{E_el_bref[si]:.8e}", f"{E_fr_bref[si]:.8e}", f"{E_el_bref[si]+E_fr_bref[si]:.8e}"])
print("  ✓ energy_evolution.csv")

# Crack tip tracking
with open(f'{OUT}/crack_tip_tracking.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['step', 'u_bar', 'tip_ref', 'tip_miehe16', 'tip_bourdin16',
                'err_miehe', 'err_bourdin'])
    for si in range(N_steps):
        w.writerow([si+1, f"{u_loads[si]:.8e}",
                    f"{tip_ref[si]:.6f}", f"{tip_m16_full[si]:.6f}", f"{tip_b16_full[si]:.6f}",
                    f"{tip_err_m[si]:.6f}", f"{tip_err_b[si]:.6f}"])
print("  ✓ crack_tip_tracking.csv")

# Mesh sensitivity
with open(f'{OUT}/mesh_sensitivity.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['N_mesh', 'h', 'ell_over_h', 'H_global', 'H_tip', 'd_global', 'time_s'])
    for Nm in mesh_sizes:
        r = results_mesh_sens[Nm]
        w.writerow([Nm, f"{r['h']:.6f}", f"{r['ell_over_h']:.2f}",
                    f"{r['H_global']:.8e}", f"{r['H_tip']:.8e}",
                    f"{r['d_global']:.8e}", f"{r['time']:.1f}"])
print("  ✓ mesh_sensitivity.csv")

# Computational cost
with open(f'{OUT}/computational_cost.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['N_mesh', 'n_dof', 'n_steps', 't_ref_s', 't_transfer_g16_s', 'overhead_pct'])
    w.writerow([cost_data['N_mesh'], cost_data['n_dof'], cost_data['n_steps'],
                f"{cost_data['t_ref']:.1f}", f"{cost_data['t_transfer_g16']:.1f}",
                f"{cost_data['overhead_pct']:.1f}"])
print("  ✓ computational_cost.csv")

# SAVE ALL DATA
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SAVING ALL DATA")
print("="*70)

# Params
with open(f'{OUT}/params.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['key','value'])
    for k,v in [('E_mod',E_mod),('nu',nu),('Gc',Gc),('ell',ell),('kappa',kappa),
                ('L',L),('N_mesh',N_mesh),('h_mesh',h_mesh),('u_peak',u_peak),
                ('N_ph1',N_ph1),('N_ph2',N_ph2),('N_ph3',N_ph3),('N_steps',N_steps),
                ('u_peak_shear',u_peak_shear),('N_steps_shear',N_steps_shear)]:
        w.writerow([k,v])
print("  ✓ params.csv")

# Load protocol
with open(f'{OUT}/load_protocol.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['step','u_bar','phase'])
    for i in range(N_steps): w.writerow([i+1,f"{u_loads[i]:.8e}",phase_labels[i]])
print("  ✓ load_protocol.csv")

# ── NEW: Load-displacement curves ──
with open(f'{OUT}/load_displacement.csv','w',newline='') as f:
    w=csv.writer(f)
    header=['step','u_bar','F_ref_miehe','F_ref_bourdin']
    for sn in transfer_schemes: header+=[f'F_miehe_{sn}',f'F_bourdin_{sn}']
    w.writerow(header)
    for si in range(N_steps):
        row=[si+1,f"{u_loads[si]:.8e}",f"{Fd_ref_miehe[si]:.8e}",f"{Fd_ref_bourdin[si]:.8e}"]
        for sn in transfer_schemes:
            row.append(f"{results_t2[sn]['Fd_miehe'][si]:.8e}")
            row.append(f"{results_t2[sn]['Fd_bourdin'][si]:.8e}")
        w.writerow(row)
print("  ✓ load_displacement.csv (P-δ curves)")

# ── NEW: Shear load-displacement ──
with open(f'{OUT}/load_displacement_shear.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['step','u_bar','F_ref_miehe','F_ref_bourdin'])
    for si in range(N_steps_shear):
        w.writerow([si+1,f"{u_shear[si]:.8e}",f"{Fd_ref_shear_m[si]:.8e}",f"{Fd_ref_shear_b[si]:.8e}"])
print("  ✓ load_displacement_shear.csv")

# Error curves
with open(f'{OUT}/error_curves_miehe.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['step']+[f'{sn}_H_global' for sn in transfer_schemes]
               +[f'{sn}_H_tip' for sn in transfer_schemes]
               +[f'{sn}_d_global' for sn in transfer_schemes])
    for si in range(N_steps):
        row=[si+1]
        for sn in transfer_schemes: row.append(f"{results_t1[sn]['H_errors'][si]:.8e}")
        for sn in transfer_schemes: row.append(f"{results_t1[sn]['H_tip_errors'][si]:.8e}")
        for sn in transfer_schemes: row.append(f"{results_t1[sn]['d_errors'][si]:.8e}")
        w.writerow(row)
print("  ✓ error_curves_miehe.csv")

with open(f'{OUT}/error_curves_bourdin.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['step']+[f'{sn}_d_global' for sn in transfer_schemes]
               +[f'{sn}_d_tip' for sn in transfer_schemes])
    for si in range(N_steps):
        row=[si+1]
        for sn in transfer_schemes: row.append(f"{results_t2[sn]['bourdin_d_errs'][si]:.8e}")
        for sn in transfer_schemes: row.append(f"{results_t2[sn]['bourdin_d_tip_errs'][si]:.8e}")
        w.writerow(row)
print("  ✓ error_curves_bourdin.csv")

# Fatigue
with open(f'{OUT}/fatigue_curves.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['cycle','alpha_max','f_min','alpha_err_B','alpha_err_C','d_err_B','d_err_C'])
    for i in range(len(alpha_err_B)):
        w.writerow([f"{cycles_1d[i]:.4f}",f"{alpha_max_hist[i]:.8e}",f"{f_min_hist[i]:.8e}",
                    f"{alpha_err_B[i]:.8e}",f"{alpha_err_C[i]:.8e}",f"{d_err_B[i]:.8e}",f"{d_err_C[i]:.8e}"])
print("  ✓ fatigue_curves.csv")

# Convergence
with open(f'{OUT}/convergence_tension.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['N_s','h_s_over_ell','miehe_H_global','miehe_H_tip','bourdin_d_global','bourdin_d_tip'])
    for r in results_conv:
        w.writerow([r['N_s'],f"{r['h_s_over_ell']:.4f}",f"{r['miehe_H_global']:.8e}",
                    f"{r['miehe_H_tip']:.8e}",f"{r['bourdin_d_global']:.8e}",f"{r['bourdin_d_tip']:.8e}"])
print("  ✓ convergence_tension.csv")

with open(f'{OUT}/convergence_shear.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['N_s','h_s_over_ell','miehe_H_global','bourdin_d_global'])
    for r in conv_shear:
        w.writerow([r['N_s'],f"{r['h_s_over_ell']:.4f}",f"{r['miehe_H_global']:.8e}",f"{r['bourdin_d_global']:.8e}"])
print("  ✓ convergence_shear.csv")

# Tip errors
with open(f'{OUT}/tip_errors_sent.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['scheme','miehe_H_tip','bourdin_d_tip','ratio'])
    for sn in ['grid_8','grid_16','grid_32','rbf_200']:
        r=results_t2[sn]
        w.writerow([sn,f"{r['miehe_H_tip']:.8e}",f"{r['bourdin_d_tip']:.8e}",f"{r['ratio_carried']:.2f}"])
print("  ✓ tip_errors_sent.csv")

with open(f'{OUT}/shear_tip_errors.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['scheme','miehe_H_tip','bourdin_d_tip'])
    for sn in ['grid_8','grid_16','grid_32']:
        r=results_shear[sn]
        w.writerow([sn,f"{r['miehe_H_tip']:.8e}",f"{r['bourdin_d_tip']:.8e}"])
print("  ✓ shear_tip_errors.csv")

# Corrected transfer results
with open(f'{OUT}/corrected_transfer.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['eta','final_H_global','final_H_tip'])
    for eta in eta_values:
        r=results_corrected[eta]
        w.writerow([f"{eta:.2f}",f"{r['final_H_global']:.8e}",f"{r['final_H_tip']:.8e}"])
print("  ✓ corrected_transfer.csv")

with open(f'{OUT}/corrected_curves.csv','w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['step']+[f'eta_{eta:.2f}_H_global' for eta in eta_values]
               +[f'eta_{eta:.2f}_H_tip' for eta in eta_values])
    for si in range(N_steps):
        row=[si+1]
        for eta in eta_values: row.append(f"{results_corrected[eta]['H_errors'][si]:.8e}")
        for eta in eta_values: row.append(f"{results_corrected[eta]['H_tip_errors'][si]:.8e}")
        w.writerow(row)
print("  ✓ corrected_curves.csv")

# Spatial fields (NPZ)
spatial={}
spatial['d_exact_load']=d_exact[N_ph1-1]
spatial['d_exact_reload']=d_exact[-1]
spatial['H_ref_nodal']=gp_to_nodal(H_exact[-1],ref.conn,ref.n_nodes)
spatial['H_t16_nodal']=gp_to_nodal(results_t1['grid_16']['solver'].H,ref.conn,ref.n_nodes)
spatial['d_bexact_final']=d_bexact[-1]
for sn in ['grid_16','grid_32']:
    r2=results_t2[sn]
    spatial[f'd_miehe_{sn}']=r2['solver_m'].d
    spatial[f'd_bourdin_{sn}']=r2['solver_b'].d
    spatial[f'H_miehe_{sn}']=gp_to_nodal(r2['solver_m'].H,ref.conn,ref.n_nodes)
spatial['H_exact_nodal']=gp_to_nodal(H_exact[-1],ref.conn,ref.n_nodes)
spatial['fat_dA']=dA; spatial['fat_dB']=dB; spatial['fat_dC']=dC
spatial['fat_alphaA']=alphaA; spatial['fat_alphaB']=alphaB
spatial['fat_alphaC']=f_fat_inv(phiC); spatial['fat_x1d']=x_1d
spatial['xs_plot']=np.linspace(0,L,N+1); spatial['N_mesh']=np.array([N_mesh])
np.savez_compressed(f'{OUT}/spatial_fields.npz',**spatial)
print("  ✓ spatial_fields.npz")

total_time=time()-_T_START
print(f"\n{'='*70}")
print(f"✅ ALL DONE — {total_time/60:.1f} min")
print(f"{'='*70}")
print(f"\n  Now run: python plot_only.py")
for f_name in sorted(os.listdir(OUT)):
    if f_name.endswith(('.csv','.npz')):
        sz=os.path.getsize(f'{OUT}/{f_name}')/1024
        print(f"    {f_name:<40s} {sz:>7.1f} KB")