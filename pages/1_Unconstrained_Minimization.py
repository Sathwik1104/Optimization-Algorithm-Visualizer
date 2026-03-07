"""
App 1 — Unconstrained Minimization
Compares Steepest Descent, Newton's Method, Conjugate Gradient
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

st.set_page_config(page_title="Unconstrained Minimization", layout="wide")

st.title("Unconstrained Minimization")
st.markdown("Compare **Steepest Descent (SD)**, **Newton's Method**, and **Conjugate Gradient (CG)** on any 2D function.")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    func_str = st.text_input("Function f(x, y)", value="5*x**2 + 4*x*y + 3*y**2",
                              help="Use x and y as variables. Python/SymPy syntax.")
    col1, col2 = st.columns(2)
    x0 = col1.number_input("Start x", value=2.0, step=0.5)
    y0 = col2.number_input("Start y", value=2.0, step=0.5)
    tol      = st.select_slider("Tolerance", options=[1e-3,1e-4,1e-5,1e-6], value=1e-6)
    max_iter = st.slider("Max iterations", 10, 500, 100)
    x_lo, x_hi = st.slider("Plot x range", -10.0, 10.0, (-3.0, 3.0))
    y_lo, y_hi = st.slider("Plot y range", -10.0, 10.0, (-3.0, 3.0))
    run_btn = st.button("Run", type="primary", use_container_width=True)

st.markdown("---")

# ── Core math ─────────────────────────────────────────────────────────────────
def build_functions(func_str):
    x_s, y_s = sp.symbols("x y")
    f_sym     = sp.sympify(func_str)
    grad_sym  = [sp.diff(f_sym, v) for v in (x_s, y_s)]
    hess_sym  = [[sp.diff(g, v) for v in (x_s, y_s)] for g in grad_sym]
    f_num     = sp.lambdify((x_s, y_s), f_sym,    "numpy")
    grad_num  = sp.lambdify((x_s, y_s), grad_sym, "numpy")
    hess_num  = sp.lambdify((x_s, y_s), hess_sym, "numpy")
    f    = lambda v: float(f_num(v[0], v[1]))
    grad = lambda v: np.array(grad_num(v[0], v[1]), dtype=float).flatten()
    hess = lambda v: np.array(hess_num(v[0], v[1]), dtype=float)
    # theoretical minimum
    try:
        sol = sp.solve(grad_sym, (x_s, y_s))
        if isinstance(sol, dict):
            min_pt = [float(sol[x_s]), float(sol[y_s])]
        else:
            min_pt = [float(sol[0][0]), float(sol[0][1])]
    except Exception:
        min_pt = None
    return f, grad, hess, f_num, min_pt

def backtrack(f, x, d, g, c=1e-4, rho=0.5):
    alpha = 1.0
    fx = f(x)
    dg = np.dot(g, d)
    while f(x + alpha * d) > fx + c * alpha * dg:
        alpha *= rho
        if alpha < 1e-12:
            break
    return alpha

def run_sd(f, grad, start, tol, max_iter):
    x, path = start.copy(), [start.copy()]
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        d = -g
        alpha = backtrack(f, x, d, g)
        x = x + alpha * d
        path.append(x.copy())
    return np.array(path)

def run_newton(f, grad, hess, start, tol, max_iter):
    x, path = start.copy(), [start.copy()]
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            d = -g
        alpha = backtrack(f, x, d, g)
        x = x + alpha * d
        path.append(x.copy())
    return np.array(path)

def run_cg(f, grad, start, tol, max_iter):
    x, path = start.copy(), [start.copy()]
    g = grad(x)
    d = -g.copy()
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        alpha   = backtrack(f, x, d, g)
        x_new   = x + alpha * d
        g_new   = grad(x_new)
        denom   = np.dot(g, g)
        beta    = max(0.0, np.dot(g_new, g_new) / denom) if denom > 1e-14 else 0.0
        d       = -g_new + beta * d
        x, g    = x_new, g_new
        path.append(x.copy())
    return np.array(path)

# ── Run & display ─────────────────────────────────────────────────────────────
if run_btn:
    start = np.array([x0, y0])
    try:
        f, grad, hess, f_num, min_pt = build_functions(func_str)
    except Exception as e:
        st.error(f"Could not parse function: {e}")
        st.stop()

    with st.spinner("Running optimizers..."):
        paths = {
            "Steepest Descent": run_sd(f, grad, start, tol, max_iter),
            "Newton":           run_newton(f, grad, hess, start, tol, max_iter),
            "Conjugate Gradient": run_cg(f, grad, start, tol, max_iter),
        }

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader("Results Summary")
    cols = st.columns(3)
    colors = {"Steepest Descent": "#e74c3c", "Newton": "#3498db", "Conjugate Gradient": "#2ecc71"}
    for col, (name, path) in zip(cols, paths.items()):
        fx_final = f(path[-1])
        col.metric(label=name, value=f"f = {fx_final:.6f}",
                   delta=f"{len(path)-1} iterations")
        col.caption(f"x = {path[-1][0]:.4f}, y = {path[-1][1]:.4f}")

    # ── Convergence plot ──────────────────────────────────────────────────────
    st.subheader("Convergence (||gradient|| over iterations)")
    fig_conv, ax_conv = plt.subplots(figsize=(10, 3))
    for name, path in paths.items():
        norms = [np.linalg.norm(grad(p)) for p in path]
        ax_conv.semilogy(norms, label=name, color=colors[name], linewidth=2)
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("||gradient||")
    ax_conv.legend()
    ax_conv.grid(True, alpha=0.3)
    st.pyplot(fig_conv, use_container_width=True)
    plt.close()

    # ── Contour + trajectory plot ─────────────────────────────────────────────
    st.subheader("Optimization Trajectories")
    xs = np.linspace(x_lo, x_hi, 200)
    ys = np.linspace(y_lo, y_hi, 200)
    X, Y = np.meshgrid(xs, ys)
    try:
        Z = f_num(X, Y)
        Z = np.where(np.isfinite(Z), Z, np.nan)
    except Exception:
        st.warning("Could not evaluate function over grid.")
        st.stop()

    fig, ax = plt.subplots(figsize=(10, 7))
    lvls = np.nanpercentile(Z, np.linspace(5, 95, 20))
    lvls = np.unique(lvls)
    if len(lvls) >= 2:
        cs = ax.contourf(X, Y, Z, levels=lvls, cmap="viridis", alpha=0.6)
        ax.contour(X, Y, Z, levels=lvls, colors="white", linewidths=0.5, alpha=0.4)
        plt.colorbar(cs, ax=ax, label="f(x, y)")

    markers = {"Steepest Descent": ("o-", "#e74c3c"),
               "Newton":           ("x--", "#3498db"),
               "Conjugate Gradient": ("*-", "#2ecc71")}
    for name, path in paths.items():
        mk, col = markers[name]
        ax.plot(path[:,0], path[:,1], mk, color=col,
                label=f"{name} ({len(path)-1} iters)", alpha=0.8, markersize=5)
    ax.plot(x0, y0, "ks", markersize=10, label="Start")
    if min_pt:
        ax.plot(min_pt[0], min_pt[1], "mX", markersize=14,
                label=f"Minimum ({min_pt[0]:.2f}, {min_pt[1]:.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"f(x,y) = {func_str}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Step-by-step table ────────────────────────────────────────────────────
    with st.expander("Show iteration details"):
        method_choice = st.selectbox("Select method", list(paths.keys()))
        path = paths[method_choice]
        rows = [{"Iter": i, "x": f"{p[0]:.6f}", "y": f"{p[1]:.6f}",
                 "f(x,y)": f"{f(p):.6f}", "||grad||": f"{np.linalg.norm(grad(p)):.2e}"}
                for i, p in enumerate(path)]
        st.dataframe(rows, use_container_width=True)
else:
    st.info("Configure the settings in the sidebar and click **Run**.")
    st.markdown("""
    **Try these functions:**
    | Function | Description |
    |---|---|
    | `5*x**2 + 4*x*y + 3*y**2` | Quadratic bowl (default) |
    | `(1-x)**2 + 100*(y-x**2)**2` | Rosenbrock (banana) — challenging |
    | `x**2 + 10*y**2` | Elongated ellipse |
    | `x**4 + y**4 - 4*x*y` | Non-convex |
    """)
